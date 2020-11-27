#include <iostream>
#include "Python.h"
#include "trainer_wrapper.hpp"
#include <stdexcept>

namespace CRFSuiteWrapper
{
    void Trainer::set_handler(PyObject *obj, messagefunc handler)
    {
        this->m_obj = obj;
        this->handler = handler;
    }

    void Trainer::message(const std::string &msg)
    {
        if (this->m_obj == NULL)
        {
            std::cerr << "** Trainer invalid state: obj [" << this->m_obj << "]\n";
            return;
        }
        PyObject *result = handler(this->m_obj, msg);
        if (result == NULL)
        {
            throw std::runtime_error("AAAaaahhhhHHhh!!!!!");
        }
    }

    void Trainer::_init_trainer()
    {
        Trainer::init();
    }
} // namespace CRFSuiteWrapper
