#ifndef TAGGER_WRAPPER_H
#define TAGGER_WRAPPER_H 1

#include <stdio.h>
#include <stdexcept>
#include "crfsuite_api.hpp"

namespace CRFSuiteWrapper
{
    class Tagger : public CRFSuite::Tagger
    {
    public:
        void dump_states(int fileno)
        {
            ensure_open();
            dump(fileno, model->dump_states);
        }

        void dump_transitions(int fileno)
        {
            ensure_open();
            dump(fileno, model->dump_transitions);
        }

    private:
        void ensure_open()
        {
            if (model == nullptr)
            {
                throw std::runtime_error("Tagger is closed");
            }
        }

        // takes ownership of the file descriptor and closes it
        void dump(int fileno, int (*dumper)(crfsuite_model_t *, FILE *))
        {
            FILE *file = fdopen(fileno, "w");
            if (!file)
            {
                throw std::runtime_error("Cannot open file");
            }

            dumper(model, file);

            if (fclose(file))
            {
                throw std::runtime_error("Cannot close file");
            }
        }
    };
} // namespace CRFSuiteWrapper
#endif
