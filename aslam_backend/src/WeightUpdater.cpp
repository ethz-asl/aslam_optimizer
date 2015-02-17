#include <aslam/backend/WeightUpdater.hpp>

namespace aslam {
	namespace backend{
		EMWeightUpdater::EMWeightUpdater(MatrixOfErrorTerms_Ptr error_terms, double lambda){
			error_terms_ = error_terms;
			scaling_factor_ = -lambda/2;
		}

		void EMWeightUpdater::updateWeights(){
			BOOST_FOREACH(VectorOfErrorTerms_Ptr vect, *error_terms_){
				double norm_factor = 0;
				std::vector<double> weights;
				BOOST_FOREACH(ErrorTerm_Ptr error_term, *vect){
					// Update weights
					double new_weight = exp(scaling_factor_*(error_term->getRawSquaredError()));
					norm_factor+= new_weight;
					weights.push_back(new_weight);
				}

				for(uint i = 0; i< vect->size();i++){
					boost::shared_ptr<aslam::backend::FixedWeightMEstimator> new_policy(new aslam::backend::FixedWeightMEstimator(weights[i]/norm_factor));
					(*vect)[i]->setMEstimatorPolicy(new_policy);
				}

			}
		}
	}
}