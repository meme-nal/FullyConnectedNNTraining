#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <torch/torch.h>

using namespace torch::indexing;

struct Options {
  torch::Device device = torch::kCPU;
  std::string trainDatasetPath = "data/train_data.csv";
  std::string testDatasetPath = "data/test_data.csv";
  size_t epochs = 500;
  size_t features_count = 4;
  size_t trainBatchSize = 20;
  size_t testBatchSize = 5;
  float lr = 0.001f;
};
static Options options;

class IrisDataset : public torch::data::Dataset<IrisDataset> {
private:
  std::vector<std::vector<float>> _features;
  std::vector<std::vector<float>> _labels;

public:
  IrisDataset(const std::string& csv_file) {
    load_csv(csv_file);
  }
  
  void load_csv(const std::string& csv_file) {
    std::ifstream file(csv_file);
    std::string line;

    if (std::getline(file, line)) {
      // The header is read and ignored
    }

    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string value;
      std::vector<float> features;
      std::vector<float> labels;

      size_t f_i {0};
      while (std::getline(ss, value, ',')) {
        if (f_i < options.features_count) {
          features.push_back(std::stof(value));
        } else {
          labels.push_back(std::stof(value));
        }
        ++f_i;
      }

      _features.push_back(features);
      _labels.push_back(labels);
    }
  }

  torch::data::Example<> get(size_t index) override {
    if (index >= _features.size()) {
      throw std::out_of_range("Index is out of bounds");
    }
    return {torch::tensor(_features.at(index)),
            torch::tensor(_labels.at(index))};
  }



  std::optional<size_t> size() const override {
    return _features.size();
  }
};


class FullyConnectedNet : public torch::nn::Module {
public:
  torch::nn::Linear fc1 {nullptr};
  torch::nn::Linear fc2 {nullptr};
  torch::nn::Linear fc3 {nullptr};  

public:
  FullyConnectedNet() {
    fc1 = register_module("fc1", torch::nn::Linear(options.features_count, 100));
    fc2 = register_module("fc2", torch::nn::Linear(100, 50));
    fc3 = register_module("fc3", torch::nn::Linear(50, 3));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = fc3->forward(x);
    return x;
  }
};


int main() {
  #ifdef DEBUG_MODE
  std::cout << "DEBUG_MODE ACTIVATED\n";
  #endif

  std::shared_ptr<FullyConnectedNet> model = std::make_shared<FullyConnectedNet>();
  model->to(options.device);
  
  auto trainDataset = IrisDataset(options.trainDatasetPath).map(torch::data::transforms::Stack<>());
  auto testDataset = IrisDataset(options.testDatasetPath).map(torch::data::transforms::Stack<>());

  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(trainDataset), options.trainBatchSize);
  auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(testDataset), options.testBatchSize);

  torch::optim::SGD optimizer(model->parameters(), options.lr);
  torch::nn::MSELoss criterion;

  for (size_t epoch {0}; epoch < options.epochs; ++epoch) {
    float batch_loss {0.f};
    size_t num_batches {0};
    for (auto& batch : *train_loader) {
      torch::Tensor features = batch.data;
      torch::Tensor labels = batch.target;

      optimizer.zero_grad();
      torch::Tensor prediction = model->forward(features.to(options.device));
      torch::Tensor loss = criterion->forward(prediction.to(options.device), labels.to(options.device));
      batch_loss += loss.item<float>();

      loss.backward();
      optimizer.step();
      ++num_batches;
    }
    
    if (epoch % 5 == 0) {
      std::cout << "Epoch: " << epoch << ", Loss: " << (batch_loss / num_batches) << std::endl;
    }    
  }

  int correct {0};
  int total {0};

  model->eval();
  torch::NoGradGuard no_grad;
  for (auto& batch : *test_loader) {
    torch::Tensor features = batch.data;
    torch::Tensor labels = batch.target.argmax(1);

    torch::Tensor prediction = model->forward(features).argmax(1);

    correct += prediction.eq(labels).sum().item<int64_t>();
    total += labels.size(0);
  }

  float accuracy = static_cast<float>(correct) / total * 100.0f;
  std::cout << "Accuracy: " << accuracy << std::endl;

  return 0;
}