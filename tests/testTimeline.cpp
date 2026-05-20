#include <iostream>
#include "../EKF.hpp"

using namespace ekf;
using namespace Eigen;

int main() {
    size_t capacity = 5;
    std::vector<int> times = {6, 7, 2, 9, 4, 11, 5, 1, 3, 8, 11, 14, 15, 16, 19, 9, 10, 17, 12, 13, 18, 20};
    std::vector<int> testTimes;
    Timeline<int> timeline(capacity);
    for(auto t:times){
        timeline.insert(t);
        testTimes.push_back(t);
        std::sort(testTimes.begin(), testTimes.end());
        while(testTimes.size() > capacity){
            testTimes.erase(testTimes.begin());
        }
        for(size_t i=timeline.head, j=0; i!=timeline.tail; i = timeline.next(i), j++){
            if(timeline[i] != testTimes[j]){
                std::cout << "Error at time " << t << ": expected " << testTimes[j] << ", got " << timeline[i] << std::endl;
                return -1;
            }
        }
    }
}
