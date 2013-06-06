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
#include <boost/shared_ptr.hpp>

namespace aslam {
namespace backend {

// PTF: As we talked about, this should be a function (unless it needs to hold state in order to be efficient)
class Marginalizer {
public:
	Marginalizer();
	virtual ~Marginalizer();

    // PTF: The function should have some comments about usage (doxygen style)
    //      This is especially important here because, as I read your code,
    //      the ordering of the design variables really matters. it is important
    //      to document stuff like this.
	void operator () (
			std::vector<aslam::backend::DesignVariable*>& inDesignVariables,
			std::vector<aslam::backend::ErrorTerm*>& inErrorTerms,
			int numberOfInputDesignVariablesToRemove,
			boost::shared_ptr<aslam::backend::MarginalizationPriorErrorTerm>& outPriorErrorTermPtr) const;

	int bla() {return 1;}
};

} /* namespace backend */
} /* namespace aslam */
#endif /* MARGINALIZER_H_ */
