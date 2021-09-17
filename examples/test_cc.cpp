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

#include <string>
#include <vector>

#include "boost/tokenizer.hpp"
#include "core/engine.hpp"
#include "io/input/inputformat_store.hpp"
#include "lib/aggregator_factory.hpp"

namespace std {
template <typename T1, typename T2>
struct hash<std::pair<T1, T2>> {
    std::size_t operator()(std::pair<T1, T2> const& p) const {
        return std::hash<T1>()(p.first) ^ std::hash<T2>()(p.second);
    }
};
}  // namespace std

using vtx_t = uint64_t;
vtx_t parse_vertex(const std::string& str) { return std::stoul(str.c_str()); }

class Vertex {
   public:
    using KeyT = vtx_t;

    Vertex() = default;
    explicit Vertex(const KeyT& id) : vertex_id(id), cid(id) {}
    const KeyT& id() const { return vertex_id; }

    // Serialization and deserialization
    friend husky::BinStream& operator<<(husky::BinStream& stream, const Vertex& v) {
        stream << v.vertex_id << v.adj << v.cid;
        return stream;
    }
    friend husky::BinStream& operator>>(husky::BinStream& stream, Vertex& v) {
        stream >> v.vertex_id >> v.adj >> v.cid;
        return stream;
    }

    vtx_t vertex_id;
    std::vector<vtx_t> adj;
    vtx_t cid = -1;
};

class Edge {
   public:
    using KeyT = std::pair<vtx_t, vtx_t>;

    Edge() = default;
    Edge(vtx_t src, vtx_t dst) : src(src), dst(dst) {}
    explicit Edge(const KeyT& id) : Edge(id.first, id.second) {}
    KeyT id() const { return {src, dst}; }

    vtx_t src, dst;
};

class Word {
   public:
    using KeyT = std::string;

    Word() = default;
    Word(const KeyT& w) : word(w) {}
    Word(KeyT&& w) : word(std::move(w)) {}
    const KeyT& id() const { return word; }

    KeyT word;
};

void cc() {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto& infmt = husky::io::InputFormatStore::create_line_inputformat();
    infmt.set_input(husky::Context::get_param("input"));

    if (husky::Context::get_global_tid() == 0)
        husky::LOG_I << "start CC";

    // Create and globalize vertex objects
    auto& edge_list = husky::ObjListStore::create_objlist<Word>();
    auto parse_edge_list = [&edge_list](boost::string_ref& chunk) {
        edge_list.add_object(std::move(chunk.to_string()));
    };
    husky::load(infmt, parse_edge_list);

    if (husky::Context::get_global_tid() == 0)
        husky::LOG_I << "read edge list finished";

    auto& vertex_list = husky::ObjListStore::create_objlist<Vertex>();
    auto& reduce_neighbors_ch = husky::ChannelStore::create_push_channel<vtx_t>(edge_list, vertex_list);
    husky::list_execute(edge_list, {}, {&reduce_neighbors_ch}, [&reduce_neighbors_ch](Word& w) {
        auto& chunk = w.word;
        if (chunk.size() == 0)
            return;
        boost::char_separator<char> sep(" \t");
        boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
        auto it = tok.begin();
        vtx_t src = parse_vertex(*it++), dst = parse_vertex(*it++);
        reduce_neighbors_ch.push(src, dst);
        reduce_neighbors_ch.push(dst, src);
    });
    husky::list_execute(vertex_list, {&reduce_neighbors_ch}, {}, [&reduce_neighbors_ch](Vertex& v) {
        const auto& msgs = reduce_neighbors_ch.get(v);
        v.adj.reserve(v.adj.size() + msgs.size());
        for (auto dst : msgs)
            v.adj.push_back(dst);
    });
    husky::globalize(vertex_list);

    if (husky::Context::get_global_tid() == 0)
        husky::LOG_I << "construct vertices finished";

    auto& ch =
        husky::ChannelStore::create_push_combined_channel<vtx_t, husky::MinCombiner<vtx_t>>(vertex_list, vertex_list);
    // Aggregator to check how many vertexes updating
    husky::lib::Aggregator<vtx_t> not_finished(0, [](vtx_t& a, const vtx_t& b) { a += b; });
    not_finished.to_reset_each_iter();

    auto& agg_ch = husky::lib::AggregatorFactory::get_channel();

    if (husky::Context::get_global_tid() == 0)
        husky::LOG_I << "Start CC";
    auto begin = std::chrono::high_resolution_clock::now();
    constexpr int repeat = 1;
    for (int i = 0; i < repeat; ++i) {
        // Initialization
        husky::list_execute(vertex_list, {}, {&ch, &agg_ch}, [&ch, &not_finished](Vertex& v) {
            not_finished.update(1);
            v.cid = v.vertex_id;
            // Get the smallest component id among neighbors
            for (auto nb : v.adj) {
                if (nb < v.cid)
                    v.cid = nb;
            }
            // Broadcast my component id
            for (auto nb : v.adj) {
                if (nb > v.cid)
                    ch.push(v.cid, nb);
            }
        });
        if (husky::Context::get_global_tid() == 0)
            husky::LOG_I << "Run " << i << " starts";
        // Main Loop
        while (not_finished.get_value()) {
            husky::list_execute(vertex_list, {&ch}, {&ch, &agg_ch}, [&ch, &not_finished](Vertex& v) {
                if (ch.has_msgs(v)) {
                    auto msg = ch.get(v);
                    if (msg < v.cid) {
                        v.cid = msg;
                        not_finished.update(1);
                        for (auto nb : v.adj)
                            ch.push(v.cid, nb);
                    }
                }
            });
            if (husky::Context::get_global_tid() == 0)
                husky::LOG_I << "# updated in this round: " << not_finished.get_value();
        }
        if (husky::Context::get_global_tid() == 0)
            husky::LOG_I << "Run " << i << " done";
    }

    auto end = std::chrono::high_resolution_clock::now();

    if (husky::Context::get_global_tid() == 0)
        husky::LOG_I << "Elapsed time (CC - " << husky::Context::get_num_global_workers()
                     << " workers): " << std::chrono::duration<double>(end - begin).count() / repeat << " sec(s)"
                     << " " << std::chrono::duration<double>(end - t1).count() << " sec(s) ";

    std::string small_graph = husky::Context::get_param("print");
    if (small_graph == "1") {
        husky::list_execute(vertex_list,
                            [](Vertex& v) { husky::LOG_I << "vertex: " << v.id() << " component id: " << v.cid; });
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("input");
    args.push_back("print");
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(cc);
        return 0;
    }
    return 1;
}
