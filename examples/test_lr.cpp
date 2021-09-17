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

#include "core/engine.hpp"
#include "lib/ml/data_loader.hpp"
#include "lib/ml/fgd.hpp"
#include "lib/ml/logistic_regression.hpp"

using husky::lib::ml::ParameterBucket;

template <typename FeatureT, typename LabelT, bool is_sparse>
void load_data(std::string url, husky::ObjList<husky::lib::ml::LabeledPointHObj<FeatureT, LabelT, is_sparse>>& data,
               const char* splitter, int num_features) {
    using DataObj = husky::lib::ml::LabeledPointHObj<FeatureT, LabelT, is_sparse>;

    auto& infmt = husky::io::InputFormatStore::create_line_inputformat();
    infmt.set_input(url);

    husky::load(infmt, [&](boost::string_ref chunk) {
        if (chunk.empty())
            return;
        boost::char_separator<char> sep(splitter);
        boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
        auto iter = tok.begin();

        DataObj this_obj(num_features);
        this_obj.y = std::stod(*iter++);
        for (int i = 0; i < num_features; ++i)
            this_obj.x.set(i, std::stod(*iter++));
        data.add_object(this_obj);
    });
}

template <bool is_sparse = false>
void logistic_regression() {
    auto t1 = std::chrono::high_resolution_clock::now();
    using LabeledPointHObj = husky::lib::ml::LabeledPointHObj<double, double, is_sparse>;
    auto& train_set = husky::ObjListStore::create_objlist<LabeledPointHObj>();

    // load data
    int num_features = std::stoi(husky::Context::get_param("num_features"));
    load_data(husky::Context::get_param("train"), train_set, " ,", num_features);

    double alpha = std::stod(husky::Context::get_param("alpha"));
    int num_iter = std::stoi(husky::Context::get_param("n_iter"));

    // initialize logistic regression model
    husky::lib::ml::LogisticRegression<double, double, is_sparse, ParameterBucket<double>> lr(num_features);
    lr.report_per_round = true;  // report training error per round

    // train the model
    auto begin = std::chrono::high_resolution_clock::now();

    lr.template train<husky::lib::ml::FGD>(train_set, num_iter, alpha);

    auto end = std::chrono::high_resolution_clock::now();

    if (husky::Context::get_global_tid() == 0)
        husky::LOG_I << "Elapsed time per iteration (LR - " << husky::Context::get_num_global_workers()
                     << " workers): " << std::chrono::duration<double>(end - begin).count() / num_iter << " sec(s)"
                     << " " << std::chrono::duration<double>(end - t1).count() << " sec(s) ";
}

void init() { logistic_regression(); }

int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("train");
    args.push_back("n_iter");
    args.push_back("alpha");
    args.push_back("num_features");
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
