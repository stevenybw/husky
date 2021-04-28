// Copyright 2016 Husky Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <signal.h>

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "boost/sort/spreadsort/string_sort.hpp"
#include "boost/tokenizer.hpp"
#include "boost/utility/string_ref.hpp"
#include "core/engine.hpp"
#include "io/input/inputformat_store.hpp"
#include "lib/aggregator_factory.hpp"

class Record {
   public:
    using KeyT = std::string;

    Record() = default;
    explicit Record(const KeyT& w) : word(w) {}
    explicit Record(KeyT&& w) : word(std::move(w)) {}
    const KeyT& id() const { return word; }

    KeyT word;

    bool operator<(const Record& other) const {
        return boost::string_ref(word).substr(0, 10) < boost::string_ref(other.word).substr(0, 10);
    }

    struct get_char_length {
        const char& operator()(const Record& r, size_t i) const { return r.word[i]; }
        size_t operator()(const Record& r) const { return 10; }
    };

    template <typename It>
    static void sort(It begin, It end) {
        boost::sort::spreadsort::string_sort(begin, end, get_char_length{}, get_char_length{});
    }

    husky::BinStream& serialize(husky::BinStream& stream) const { return stream << word; }

    husky::BinStream& deserialize(husky::BinStream& stream) { return stream >> word; }
};

void wc() {
    auto& infmt = husky::io::InputFormatStore::create_line_inputformat();
    infmt.set_input(husky::Context::get_param("input"));
    auto& records_list = husky::ObjListStore::create_objlist<Record>();

    auto parse_tera = [&](boost::string_ref& chunk) {
        if (chunk.size() == 0)
            return;
        records_list.add_object(Record(chunk.to_string()));
    };

    husky::load(infmt, parse_tera);
    husky::globalize(records_list);

    husky::lib::AggregatorFactory::sync();

    if (husky::Context::get_global_tid() == 0)
        husky::LOG_I << "Start sorting.";

    auto p = husky::Context::get_num_global_workers();
    Record::sort(records_list.get_data().begin(), records_list.get_data().end());

    if (husky::Context::get_global_tid() == 0)
        husky::LOG_I << "Sorting: local sorted.";

    husky::lib::Aggregator<std::vector<Record>> samples_aggr(std::vector<Record>(),
                                                             [](std::vector<Record>& a, const std::vector<Record>& b) {
                                                                 for (auto& x : b)
                                                                     a.push_back(x);
                                                             });
    size_t stride = (records_list.get_size() + p - 1) / p;
    std::vector<Record> l_samples(p - 1);
    for (int i = 1; i < p; ++i)
        l_samples[i - 1] = records_list.get(std::min(records_list.get_size() - 1, i * stride));
    samples_aggr.update(l_samples);
    {
        husky::ChannelManager out_manager({&husky::lib::AggregatorFactory::get_channel()});
        out_manager.flush();
    }
    husky::lib::AggregatorFactory::sync();
    auto g_samples = samples_aggr.get_value();
    Record::sort(g_samples.begin(), g_samples.end());
    std::vector<Record> samples(p - 1);
    for (int i = 1; i < p; ++i)
        samples[i - 1] = g_samples[i * (p - 1)];

    if (husky::Context::get_global_tid() == 0)
        husky::LOG_I << "Sorting: regular samples selected.";

    auto& sorted_list = husky::ObjListStore::create_objlist<Record>();
    auto& migrate_channel = husky::ChannelStore::create_migrate_channel(records_list, sorted_list);
    husky::list_execute(records_list, {}, {&migrate_channel},
                        [&samples, j = 0, p, &migrate_channel](Record& r) mutable {
                            while (j < p - 1 && samples[j] < r)
                                j++;
                            migrate_channel.migrate(r, j);
                        });
    {
        husky::ChannelManager in_manager({&migrate_channel});
        in_manager.poll_and_distribute();
    }

    Record::sort(sorted_list.get_data().begin(), sorted_list.get_data().end());

    husky::lib::AggregatorFactory::sync();

    if (husky::Context::get_global_tid() == 0)
        husky::LOG_I << "Sorting done.";

    husky::lib::Aggregator<std::map<int, std::vector<std::string>>> result_collect_aggr(
        std::map<int, std::vector<std::string>>(),
        [](std::map<int, std::vector<std::string>>& a, const std::map<int, std::vector<std::string>>& b) {
            for (auto& [k, v] : b)
                a[k] = v;
        });
    result_collect_aggr.update_any([&](std::map<int, std::vector<std::string>>& x) {
        x[husky::Context::get_global_tid()] =
            std::vector{sorted_list.get(0).id(), sorted_list.get(sorted_list.get_size() - 1).id()};
    });
    {
        husky::ChannelManager out_manager({&husky::lib::AggregatorFactory::get_channel()});
        out_manager.flush();
    }
    husky::lib::AggregatorFactory::sync();
    if (husky::Context::get_global_tid() == 0) {
        for (int i = 1; i < p; ++i) {
            auto& next_begin = result_collect_aggr.get_value()[i][0];
            auto& last_end = result_collect_aggr.get_value()[i - 1][1];
            if (next_begin < last_end)
                husky::LOG_E << "Thread " << i - 1 << " ends with '" << last_end << "'; but thread " << i
                             << " starts with '" << next_begin << "'!";
        }
    }
}

void sighndl(int x) {
    char hostname[256];
    gethostname(hostname, 256);
    std::cout << "[" << hostname << ":" << getpid() << "] received signal " << x << std::endl;
    while (1)
        ;
}

int main(int argc, char** argv) {
    signal(6, sighndl);
    signal(8, sighndl);
    signal(11, sighndl);
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("input");
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(wc);
        return 0;
    }
    return 1;
}
