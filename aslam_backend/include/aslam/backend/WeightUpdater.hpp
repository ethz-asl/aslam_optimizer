#ifndef ASLAM_WEIGHT_UPDATER_HPP
#define ASLAM_WEIGHT_UPDATER_HPP

#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <math.h> 

#include <aslam/backend/ErrorTerm.hpp>

namespace aslam{
	namespace backend{

		class WeightUpdater{
		public:
			virtual void updateWeight()= 0;
		};

		// Update the weights of the error terms using a gaussian of the squared error, i.e: exp(-lambda/2*(y -f(x))^2)
		// The matrix of error terms contains, in each row, the error terms whose weights must be normalized together
		class EMWeightUpdater: public WeightUpdater{
		private:
			typedef boost::shared_ptr<ErrorTerm> ErrorTerm_Ptr;
			typedef boost::shared_ptr<std::vector<ErrorTerm_Ptr>> VectorOfErrorTerms_Ptr; 
			typedef boost::shared_ptr<std::vector<VectorOfErrorTerms_Ptr>> MatrixOfErrorTerms_Ptr;
			MatrixOfErrorTerms_Ptr error_terms_;
			double scaling_factor_;
		public:
			EMWeightUpdater(MatrixOfErrorTerms_Ptr error_terms, double lambda);
			void updateWeights();
		};
	}
}

#endif