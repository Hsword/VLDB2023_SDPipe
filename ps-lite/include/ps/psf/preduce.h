#pragma once

#include "PSFunc.h"

namespace ps {

template <>
struct PSFData<kPReduceGetPartner> {
    static constexpr PsfGroup group = PsfGroup::kPReduceScheduler;
    static constexpr const char *name = "PReduceGetPartner";
    using Request =
        tuple<Key,    // reduce group key, each pipeline stage has a unique key
              int,    // worker rank
              int,    // worker vertical rank (data parallel rank)
              size_t, // batch id (for vertical stop control)
              float   // max wait time (ms)
              >;
    using Response =
        tuple<SArray<int>, // all the partners worker id to do reduce with
              int          // whether the global sync point reaches
              >;
    static void _callback(const Response &response, int *tgt) {
        auto &val = get<0>(response);
        *tgt = get<1>(response);
        std::copy(val.begin(), val.end(), ++tgt);
        tgt[val.size()] = -1;
    }
};

template <>
struct PSFData<kPReduceInit> {
    static constexpr PsfGroup group = PsfGroup::kPReduceScheduler;
    static constexpr const char *name = "PReduceInit";
    using Request =
        tuple<Key,    // reduce group key, each pipeline stage has a unique key
              int,    // worker rank
              size_t, // number of workers
              size_t, // ssp_bound, guaruntee connection graph for some
                      // iteration
              size_t  // global_sync, synchronize all workers for some iteration
              >;
    using Response = tuple<>;
    static void _callback(const Response &response) {
    }
};

} // namespace ps
