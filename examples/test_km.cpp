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

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "boost/tokenizer.hpp"
#include "core/engine.hpp"
#include "io/input/inputformat_store.hpp"
#include "lib/aggregator_factory.hpp"
#include "lib/vector.hpp"

constexpr int LEN = 64;

struct KMeansPoint {
    using KeyT = size_t;
    size_t idx;
    Eigen::Matrix<double, LEN, 1> data;

    KMeansPoint() : idx(0) { data.setZero(); }
    explicit KMeansPoint(const KeyT& idx) : idx(idx) { data.setZero(); }
    const KeyT& id() const { return idx; }

    double distanceEuclid(const KMeansPoint& p) const { return (data - p.data).squaredNorm(); }

    int getClosetCenterId(const std::vector<KMeansPoint>& centers) const {
        double min_dis = INFINITY;
        int32_t min_id = 0;
        for (size_t i = 0; i < centers.size(); ++i) {
            auto dis = distanceEuclid(centers[i]);
            if (dis < min_dis) {
                min_dis = dis;
                min_id = i;
            }
        }
        return min_id;
    }

    husky::BinStream& serialize(husky::BinStream& stream) const {
        stream << idx;
        stream.push_back_bytes((const char*) data.data(), data.size() * sizeof(double));
        return stream;
    }
    husky::BinStream& deserialize(husky::BinStream& stream) {
        stream >> idx;
        std::copy_n((double*) stream.pop_front_bytes(data.size() * sizeof(double)), data.size(), data.data());
        return stream;
    }
};

void load_data(std::string url, husky::ObjList<KMeansPoint>& data, const char* splitter) {
    auto& infmt = husky::io::InputFormatStore::create_line_inputformat();
    infmt.set_input(url);

    husky::load(infmt, [&](boost::string_ref chunk) {
        try {
            if (chunk.empty())
                return;
            boost::char_separator<char> sep(splitter);
            boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
            auto iter = tok.begin();

            if (iter.at_end()) {
                // husky::LOG_E << "problematic input: '" << chunk << "'";
                return;
            }
            int idx = std::stoi(*iter++);
            KMeansPoint point(idx);
            for (int i = 0; i < LEN; ++i) {
                if (iter.at_end()) {
                    // husky::LOG_E << "problematic input: '" << chunk << "'";
                    return;
                }
                point.data(i) = std::stod(*iter++);
            }
            data.add_object(point);
        } catch (const std::invalid_argument& e) {
            husky::LOG_E << "error parsing '" << chunk << "'." << std::endl;
        }
    });
}

void kmeans() {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto& points = husky::ObjListStore::create_objlist<KMeansPoint>();

    // load data
    load_data(husky::Context::get_param("input"), points, " ");

    int num_iter = std::stoi(husky::Context::get_param("n_iter"));
    int K = std::stoi(husky::Context::get_param("k"));

    std::random_device rd;
    std::minstd_rand gen(rd());
    husky::lib::Aggregator<std::pair<std::vector<KMeansPoint>, size_t>> centers_init_aggr(
        std::pair<std::vector<KMeansPoint>, size_t>{std::vector<KMeansPoint>(K), 0},
        [K, &gen](std::pair<std::vector<KMeansPoint>, size_t>& a,
                  const std::pair<std::vector<KMeansPoint>, size_t>& b) {
            auto& [centers_a, count_a] = a;
            const auto& [centers_b, count_b] = b;
            std::discrete_distribution dist{(double) count_a, (double) count_b};
            for (int i = 0; i < K; ++i)
                if (dist(gen))
                    centers_a[i] = centers_b[i];
            count_a += count_b;
        },
        [K](auto& x) {
            x.first.resize(K);
            x.second = 0;
        });

    {
        std::vector<KMeansPoint> local_centers_init(K);
        if (points.get_vector_size()) {
            std::uniform_int_distribution<size_t> dist(0, points.get_vector_size() - 1);
            for (int i = 0; i < K; i++)
                local_centers_init[i] = points.get(dist(gen));
        }
        centers_init_aggr.update({std::move(local_centers_init), points.get_vector_size()});
        husky::ChannelManager out_manager(
            std::vector<husky::ChannelBase*>{&husky::lib::AggregatorFactory::get_channel()});
        out_manager.flush();
    }
    husky::lib::AggregatorFactory::sync();
    std::vector<KMeansPoint> centers;
    size_t total_count;
    std::tie(centers, total_count) = centers_init_aggr.get_value();

    using IntermediateCenters = std::vector<std::pair<KMeansPoint, size_t>>;
    auto apply_point_to_center = [&centers, K](IntermediateCenters& c, const KMeansPoint& p) {
        int id = p.getClosetCenterId(centers);
        c[id].first.data += p.data;
        c[id].second++;
    };
    husky::lib::Aggregator<IntermediateCenters> aggr(
        std::vector(K, std::pair{KMeansPoint(), 0ul}),
        [K](IntermediateCenters& a, const IntermediateCenters& b) {
            for (int i = 0; i < K; ++i) {
                a[i].first.data += b[i].first.data;
                a[i].second += b[i].second;
            }
        },
        [K](IntermediateCenters& x) {
            x.resize(K, {KMeansPoint(), 0ul});
        });
    aggr.to_reset_each_iter();

    {
        husky::lib::Aggregator wssseAggr(0.0);
        husky::list_execute(points, [&wssseAggr, &centers](KMeansPoint& p) {
            int id = p.getClosetCenterId(centers);
            wssseAggr.update(p.distanceEuclid(centers[id]));
        });
        husky::lib::AggregatorFactory::sync();
        if (husky::Context::get_global_tid() == 0)
            husky::LOG_I << "Initial WSSSE: " << wssseAggr.get_value();
    }

    if (husky::Context::get_global_tid() == 0)
        husky::LOG_I << "Sampled " << K << " initial centers from " << total_count << " points.";

    auto begin = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < num_iter; iter++) {
        husky::list_execute(points,
                            [&aggr, apply_point_to_center](KMeansPoint& p) { aggr.update(apply_point_to_center, p); });
        husky::lib::AggregatorFactory::sync();
        auto& result = aggr.get_value();
        for (int i = 0; i < K; ++i)
            centers[i].data = result[i].first.data / result[i].second;

        if (husky::Context::get_global_tid() == 0)
            husky::LOG_I << "Iteration " << iter << " done.";
    }

    auto end = std::chrono::high_resolution_clock::now();

    if (husky::Context::get_global_tid() == 0)
        husky::LOG_I << "Elapsed time per iteration (KM - " << husky::Context::get_num_global_workers()
                     << " workers): " << std::chrono::duration<double>(end - begin).count() / num_iter << " sec(s)"
                     << " " << std::chrono::duration<double>(end - t1).count() << " sec(s) ";

    {
        husky::lib::Aggregator wssseAggr(0.0);
        husky::list_execute(points, [&wssseAggr, &centers](KMeansPoint& p) {
            int id = p.getClosetCenterId(centers);
            wssseAggr.update(p.distanceEuclid(centers[id]));
        });
        husky::lib::AggregatorFactory::sync();
        if (husky::Context::get_global_tid() == 0)
            husky::LOG_I << "Result WSSSE: " << wssseAggr.get_value();
    }
}

void init() { kmeans(); }

void sighndl(int x) {
    char hostname[256];
    gethostname(hostname, 256);
    std::cerr << "[" << hostname << ":" << getpid() << "] received signal " << x << std::endl;
    while (1)
        ;
}

int main(int argc, char** argv) {
    signal(6, sighndl);
    signal(11, sighndl);
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("input");
    args.push_back("n_iter");
    args.push_back("k");
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
