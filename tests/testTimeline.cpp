#include <iostream>
#include "../EKF.hpp"

using namespace ekf;
using namespace Eigen;

int main() {
    size_t capacity = 5;
    std::vector<int> times = {6, 7, 2, 9, 4, 11, 5, 1, 3, 8, 11, 14, 15, 16, 19, 9, 10, 17, 12, 13, 18, 20};
    std::vector<int> lookupTimes = {6, 10, 5, 15, 20, 25};
    std::vector<int> testTimes;
    Timeline<int> timeline(capacity);
    for(auto t:times){
        timeline.insert(t);
        testTimes.push_back(t);
        std::sort(testTimes.begin(), testTimes.end());
        while(testTimes.size() > capacity){
            testTimes.erase(testTimes.begin());
        }
        size_t j=0;
        for(auto i=timeline.head; i!=timeline.tail; i = timeline.next(i), j++){
            if(timeline[i] != testTimes[j]){
                std::cout << "Error at time " << t
                          << ": expected " << testTimes[j]
                          << ", got " << timeline[i] << std::endl;
                return -1;
            }
        }
        for(auto lookupTime:lookupTimes){
            auto f = timeline.find(lookupTime);
            for(size_t j=testTimes.size()-1; j!=0; j--){
                if(testTimes[j] <= lookupTime){
                    if(testTimes[j] != timeline[f]){
                        std::cout << "Error at lookup of " << lookupTime
                                  << ": expected " << testTimes[j]
                                  << ", got " << timeline[f] << std::endl;
                        std::cout << "Timeline contents: ";
                        for(auto i=timeline.head; i!=timeline.tail; i = timeline.next(i)){
                            std::cout<<timeline[i] << " ";
                        }
                        std::cout << std::endl;
                        std::cout << "Expected contents: ";
                        for(auto t:testTimes){
                            std::cout << t << " ";
                        }
                        std::cout << std::endl;
                        return -1;
                    }
                    break;
                }
            }
        }
    }
}
