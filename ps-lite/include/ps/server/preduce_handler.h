#pragma once

#include "ps/psf/PSFunc.h"

#include <unordered_map>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>

namespace ps {
template<>
class PSHandler<PsfGroup::kPReduceScheduler> : public PSHandler<PsfGroup::kBaseGroup> {
public:
    PSHandler();
    ~PSHandler();
    PSHandler(const PSHandler<PsfGroup::kPReduceScheduler> &handle) = delete;

    void serve(const PSFData<kPReduceGetPartner>::Request &request,
               PSFData<kPReduceGetPartner>::Response &response);

    void serve(const PSFData<kPReduceInit>::Request &request,
               PSFData<kPReduceInit>::Response &response);

private:
    class ReduceStat;
    std::unordered_map<Key, std::unique_ptr<ReduceStat>> map_;
    std::mutex map_mtx_; // lock for the map
};

} // namespace ps
