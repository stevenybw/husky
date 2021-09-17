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

namespace std {
template <typename T1, typename T2>
struct hash<std::pair<T1, T2>> {
    std::size_t operator()(std::pair<T1, T2> const& p) const {
        return std::hash<T1>()(p.first) ^ std::hash<T2>()(p.second);
    }
};
}  // namespace std

using vtx_t = int64_t;
vtx_t parse_vertex(const std::string& str) { return std::stol(str.c_str()); }

class Vertex {
   public:
    using KeyT = vtx_t;

    Vertex() : pr(0.15) {}
    explicit Vertex(const KeyT& id) : vertex_id(id), pr(0.15) {}
    const KeyT& id() const { return vertex_id; }

    // Serialization and deserialization
    friend husky::BinStream& operator<<(husky::BinStream& stream, const Vertex& v) {
        stream << v.vertex_id << v.adj << v.pr;
        return stream;
    }
    friend husky::BinStream& operator>>(husky::BinStream& stream, Vertex& v) {
        stream >> v.vertex_id >> v.adj >> v.pr;
        return stream;
    }

    vtx_t vertex_id;
    std::vector<vtx_t> adj;
    double pr;
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

void pagerank() {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto& infmt = husky::io::InputFormatStore::create_line_inputformat();
    infmt.set_input(husky::Context::get_param("input"));

    if (husky::Context::get_global_tid() == 0)
        husky::LOG_I << "start PR";

    // Create and globalize vertex objects
    auto& edge_list = husky::ObjListStore::create_objlist<Edge>();
    auto parse_edge_list = [&edge_list](boost::string_ref& chunk) {
        if (chunk.size() == 0)
            return;
        boost::char_separator<char> sep(" \t");
        boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
        auto it = tok.begin();
        vtx_t src = parse_vertex(*it++), dst = parse_vertex(*it++);
        edge_list.add_object(Edge(src, dst));
    };
    husky::load(infmt, parse_edge_list);

    if (husky::Context::get_global_tid() == 0)
        husky::LOG_I << "read edge list finished";

    auto& vertex_list = husky::ObjListStore::create_objlist<Vertex>();
    auto& reduce_neighbors_ch = husky::ChannelStore::create_push_channel<vtx_t>(edge_list, vertex_list);
    husky::list_execute(edge_list, {}, {&reduce_neighbors_ch}, [&reduce_neighbors_ch](Edge& e) {
        reduce_neighbors_ch.push(-1, e.dst);
        reduce_neighbors_ch.push(e.dst, e.src);
    });
    husky::list_execute(vertex_list, {&reduce_neighbors_ch}, {}, [&reduce_neighbors_ch](Vertex& v) {
        const auto& msgs = reduce_neighbors_ch.get(v);
        v.adj.reserve(v.adj.size() + msgs.size());
        for (vtx_t dst : msgs)
            if (dst > 0)
                v.adj.push_back(dst);
    });
    husky::globalize(vertex_list);

    if (husky::Context::get_global_tid() == 0)
        husky::LOG_I << "construct vertices finished";

    // Iterative PageRank computation
    auto& prch =
        husky::ChannelStore::create_push_combined_channel<double, husky::SumCombiner<double>>(vertex_list, vertex_list);
    int numIters = stoi(husky::Context::get_param("iters"));

    auto begin = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < numIters; ++iter) {
        husky::list_execute(vertex_list, [&prch, iter](Vertex& u) {
            if (iter > 0)
                u.pr = 0.85 * prch.get(u) + 0.15;

            if (u.adj.size() == 0)
                return;
            double sendPR = u.pr / u.adj.size();
            for (auto& nb : u.adj) {
                prch.push(sendPR, nb);
            }
        });

        if (husky::Context::get_global_tid() == 0)
            husky::LOG_I << "iteration " << iter << " done";
    }

    auto end = std::chrono::high_resolution_clock::now();

    if (husky::Context::get_global_tid() == 0)
        husky::LOG_I << "Elapsed time per iteration (PR - " << husky::Context::get_num_global_workers()
                     << " workers): " << std::chrono::duration<double>(end - begin).count() / numIters << " sec(s)"
                     << " " << std::chrono::duration<double>(end - t1).count() << " sec(s) ";
}

int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("input");
    args.push_back("iters");
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(pagerank);
        return 0;
    }
    return 1;
}
