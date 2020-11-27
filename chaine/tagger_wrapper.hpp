#ifndef TAGGER_WRAPPER_H
#define TAGGER_WRAPPER_H 1

#include <stdio.h>
#include <errno.h>
#include <stdexcept>
#include "crfsuite_api.hpp"

namespace CRFSuiteWrapper
{
    class Tagger : public CRFSuite::Tagger
    {
    public:
        void dump(int fileno)
        {
            if (model == NULL)
            {
                throw std::runtime_error("Tagger is closed");
            }

            FILE *file = fdopen(fileno, "w");
            if (!file)
            {
                throw std::runtime_error("Cannot open file");
            }

            model->dump(model, file);

            if (fclose(file))
            {
                throw std::runtime_error("Cannot close file");
            };
        }
    };
} // namespace CRFSuiteWrapper
#endif
