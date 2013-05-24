/*
 * Marginalizer.h
 *
 *  Created on: May 24, 2013
 *      Author: mbuerki
 */

#ifndef MARGINALIZER_H_
#define MARGINALIZER_H_

#include <aslam/backend/DesignVariable.hpp>
#include <aslam/backend/MarginalizationPriorErrorTerm.hpp>
#include <aslam/backend/ErrorTerm.hpp>
//#include <boost/shared_ptr.hpp>

namespace aslam {
namespace backend {

class Marginalizer {
public:
	Marginalizer();
	virtual ~Marginalizer();
	void marginalize(
			std::vector<aslam::backend::DesignVariable*>& inDesignVariables,
			std::vector<aslam::backend::ErrorTerm*>& inErrorTerms,
			int numberOfInputDesignVariablesToRemove,
			aslam::backend::MarginalizationPriorErrorTerm** outPriorErrorTerm) const;

	int bla() {return 1;}
};

} /* namespace backend */
} /* namespace aslam */
#endif /* MARGINALIZER_H_ */
