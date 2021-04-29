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

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "boost/tokenizer.hpp"
#include "core/engine.hpp"
#include "io/input/inputformat_store.hpp"
#include "lib/aggregator_factory.hpp"

class Word {
   public:
    using KeyT = std::string;

    Word() = default;
    explicit Word(const KeyT& w) : word(w) {}
    const KeyT& id() const { return word; }

    KeyT word;
    int count = 0;
};

bool operator<(const std::pair<int, std::string>& a, const std::pair<int, std::string>& b) {
    return a.first == b.first ? a.second < b.second : a.first < b.first;
}

void wc() {
    auto& infmt = husky::io::InputFormatStore::create_line_inputformat();
    infmt.set_input(husky::Context::get_param("input"));
    auto& raw_word_list = husky::ObjListStore::create_objlist<Word>();

    auto parse_wc = [&](boost::string_ref& chunk) {
        if (chunk.size() == 0)
            return;
        boost::char_separator<char> sep(" \r\t");
        boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
        for (auto& w : tok) {
            raw_word_list.add_object(Word(w));
        }
    };

    husky::load(infmt, parse_wc);

    husky::lib::AggregatorFactory::sync();

    // Show topk words.
    const int kMaxNum = 100;
    typedef std::set<std::pair<int, std::string>> TopKPairs;
    auto add_to_topk = [](TopKPairs& pairs, const std::pair<int, std::string>& p) {
        if (pairs.size() == kMaxNum && *pairs.begin() < p)
            pairs.erase(pairs.begin());
        if (pairs.size() < kMaxNum)
            pairs.insert(p);
    };
    husky::lib::Aggregator<TopKPairs> unique_topk(
        TopKPairs(),
        [add_to_topk](TopKPairs& a, const TopKPairs& b) {
            for (auto& i : b)
                add_to_topk(a, i);
        },
        [](TopKPairs& a) { a.clear(); },
        [add_to_topk](husky::base::BinStream& in, TopKPairs& pairs) {
            pairs.clear();
            for (size_t n = husky::base::deser<size_t>(in); n--;)
                add_to_topk(pairs, husky::base::deser<std::pair<int, std::string>>(in));
        },
        [](husky::base::BinStream& out, const TopKPairs& pairs) {
            out << pairs.size();
            for (auto& p : pairs)
                out << p;
        });
    unique_topk.to_reset_each_iter();

    if (husky::Context::get_global_tid() == 0)
        husky::LOG_I << "Start WordCount";
    auto begin = std::chrono::high_resolution_clock::now();

    constexpr int repeat = 10;
    for (int i = 0; i < repeat; ++i) {
        auto& word_list = husky::ObjListStore::create_objlist<Word>("word_list");
        auto& ch =
            husky::ChannelStore::create_push_combined_channel<int, husky::SumCombiner<int>>(raw_word_list, word_list, "wc_ch");
        husky::lib::AggregatorFactory::sync();
        husky::list_execute(raw_word_list, [&](Word& s) { ch.push(1, s.word); });
        husky::list_execute(word_list, [&ch, &unique_topk, add_to_topk](Word& word) {
            unique_topk.update(add_to_topk, std::make_pair(ch.get(word), word.id()));
        });
        husky::lib::AggregatorFactory::sync();
        if (husky::Context::get_global_tid() == 0)
            husky::LOG_I << "Run " << i << " done";
        husky::ObjListStore::drop_objlist("word_list");
        husky::ChannelStore::drop_channel("wc_ch");
    }

    auto end = std::chrono::high_resolution_clock::now();

    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << "Elapsed time (WC - " << husky::Context::get_num_global_workers()
                     << " workers): " << std::chrono::duration<double>(end - begin).count() / repeat << " sec(s)";
        for (auto& i : unique_topk.get_value())
            husky::LOG_I << i.second << " " << i.first;
    }
}

int main(int argc, char** argv) {
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
