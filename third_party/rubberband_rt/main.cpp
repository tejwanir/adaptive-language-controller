#include <RubberBandStretcher.h>
#include <exception>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

struct RubberBandStretcher {
    RubberBandStretcher(std::size_t sampleRate, std::size_t channels)
        : _stretcher{sampleRate, channels,
                     RubberBand::RubberBandStretcher::PresetOption::DefaultOptions |
                         RubberBand::RubberBandStretcher::Option::OptionProcessRealTime |
                         RubberBand::RubberBandStretcher::Option::OptionEngineFiner} {}
    double getTimeRatio() const { return _stretcher.getTimeRatio(); }
    void setTimeRatio(double timeRatio) { _stretcher.setTimeRatio(timeRatio); }
    std::size_t getStartDelay() const { return _stretcher.getStartDelay(); }
    std::size_t getPreferredStartPad() const { return _stretcher.getPreferredStartPad(); }
    std::size_t getSamplesRequired() const { return _stretcher.getSamplesRequired(); }
    int available() const { return _stretcher.available(); }

    void process(py::array_t<float, py::array::c_style | py::array::forcecast> input, bool final) {
        const std::size_t channels = input.shape(0);
        if (input.ndim() != 2 || channels != _stretcher.getChannelCount())
            throw new std::length_error("invalid audio input buffer");
        const std::size_t samples = input.shape(1);
        const float *ptr = static_cast<const float *>(input.request().ptr);
        const float **ptrs = new const float *[channels];
        for (auto i = 0; i < channels; i++)
            ptrs[i] = ptr + input.index_at(i, 0);
        _stretcher.process(ptrs, samples, final);
        delete[] ptrs;
    }

    std::size_t retrieve(py::array_t<float> output) {
        const std::size_t channels = output.shape(0);
        if (output.ndim() != 2 || channels != _stretcher.getChannelCount())
            throw new std::length_error("invalid audio output buffer");
        const std::size_t samples = output.shape(1);
        float *ptr = static_cast<float *>(output.request().ptr);
        float **ptrs = new float *[channels];
        for (auto i = 0; i < channels; i++)
            ptrs[i] = ptr + output.index_at(i, 0);
        const std::size_t ret = _stretcher.retrieve(ptrs, samples);
        delete[] ptrs;
        return ret;
    }

  private:
    RubberBand::RubberBandStretcher _stretcher;
};

PYBIND11_MODULE(rubberband_rt, m) {
    m.doc() = R"pbdoc(
        Bindings for real-time audio scaling for the Rubberband library.
    )pbdoc";

    py::class_<RubberBandStretcher>(m, "RubberBandStretcher")
        .def(py::init<size_t, size_t>())
        .def("getTimeRatio", &RubberBandStretcher::getTimeRatio)
        .def("setTimeRatio", &RubberBandStretcher::setTimeRatio)
        .def("getStartDelay", &RubberBandStretcher::getStartDelay)
        .def("getPreferredStartPad", &RubberBandStretcher::getPreferredStartPad)
        .def("getSamplesRequired", &RubberBandStretcher::getSamplesRequired)
        .def("available", &RubberBandStretcher::available)
        .def("process", &RubberBandStretcher::process)
        .def("retrieve", &RubberBandStretcher::retrieve);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}