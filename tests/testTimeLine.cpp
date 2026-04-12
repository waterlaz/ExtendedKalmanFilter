#include <iostream>
#include "../EKF.hpp"

using namespace EKFNamespace;
using namespace Eigen;

int main() {
    std::vector<double> times =
        {1.0, 2.0, 3.0, 4.0, 5.0, 1.01, 5.01, 2.01, 1.02, 0.99, 0.01, 4.99, 5.02};
    //     0    1    2    3    4     5     6     7     8     9    10    11    12
    //     1    2    3    4    5     1     5     2     1     1     0     5     5
    std::vector<VectorXd> expected =
        {VectorXd(1),  // 0
         VectorXd(4),  // 1
         VectorXd(2),  // 2
         VectorXd(1),  // 3
         VectorXd(1),  // 4
         VectorXd(4)}; // 5

    expected[0] << 10;
    expected[1] << 0, 5, 8, 9;
    expected[2] << 1, 7;
    expected[3] << 2;
    expected[4] << 3;
    expected[5] << 4, 6, 11, 12;

    TimeLine<double, 1> timeline(0.1);
    for(size_t i=0; i<times.size(); i++){
        Matrix<double, 1, 1> z(i);
        Matrix<double, 1, 1> J(1);
        Matrix<double, 1, 1> C(1);
        timeline.insert(TimeStep(times[i], z, J, C));
    }
    int i = 0;
    for(auto& step : timeline){
        //std::cout<<step.measurement.transpose() << " = " << expected[i].transpose() << std::endl;
        if((step.measurement - expected[i]).norm() > 1e-6){
            std::cerr << "Test failed at index " << i << std::endl;
            return -1;
        }
        i++;
    }
}
