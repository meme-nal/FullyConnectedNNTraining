#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <torch/torch.h>


class IrisDataset : public torch::data::Dataset<IrisDataset> {
private:
  std::vector<std::pair<torch::Tensor, torch::Tensor>> _data;

public:
  IrisDataset(const std::string& csv_file) {
    load_csv(csv_file);
  }

  void load_csv(const std::string& csv_file);

  torch::data::Example<> get(size_t index) override {
    return {_data[index].first, _data[index].second};
  }

  size_t size() override {
    return _data.size();
  }
};


int main() {
  #ifdef DEBUG_MODE
  std::cout << "DEBUG_MODE ACTIVATED\n";
  #endif

  std::string trainDataPath {"data/train_data.csv"};
  std::string testDataPath {"data/test_data.csv"};

  IrisDataset trainDataset(trainDataPath);
  IrisDataset testDataset(testDataPath);

  return 0;
}