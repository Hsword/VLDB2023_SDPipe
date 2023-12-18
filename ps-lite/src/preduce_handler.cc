#include "ps/server/preduce_handler.h"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

using namespace std::chrono;

namespace ps {

// store the state for every reduce key
class PSHandler<PsfGroup::kPReduceScheduler>::ReduceStat {
public:
    ReduceStat(int key, size_t max_workers, size_t ssp_bound,
               size_t sync_every) :
        key_(key),
        max_workers_(max_workers), ssp_bound_(max_workers * ssp_bound),
        sync_every_(max_workers * sync_every) {
    }

    std::pair<std::vector<int>, bool>
    getPartner(int rank, int vertical_rank, size_t batch_id, float wait_time) {
        std::unique_lock<std::mutex> lock(mtx_);
        // must wait until the previous partial reduce decision finish
        while (critical_count)
            cv_.wait(lock);

        ready_workers_.push_back(rank);
        DisjointSetMerge(rank, ready_workers_[0]);
        if (key_ == 0 && stop_count_ > 0 && !stopped_workers_.count(rank)) {
            // when stopped, set the condition so that all worker in the pipe
            // can see the condition and stop at the same batch id
            stop_count_--;
            stopped_workers_.insert(rank);
            setStopCondition(vertical_rank, batch_id);
        }
        const bool should_stop = checkStopCondition(vertical_rank, batch_id);
        avail_workers_.insert(rank);
        if (ready_workers_.size() == 1) {
            do_graph_sync_ = checkSyncType(key_, need_graph_sync_);
            if (avail_workers_.size() < max_workers_)
                do_graph_sync_ = false;
            // the first worker should set the wait time for others
            wake_time_ =
                system_clock::now() + microseconds(int(wait_time * 1000));
        }
        if (do_graph_sync_) {
            if (DisjointSetIsFullyMerged()) {
                cv_.notify_all();
            } else {
                while (!DisjointSetIsFullyMerged())
                    cv_.wait(lock);
            }
        } else {
            if (ready_workers_.size() == max_workers_) {
                // if worker number is enough, notify all
                cv_.notify_all();
            } else {
                while (ready_workers_.size() < max_workers_
                       && cv_.wait_until(lock, wake_time_)
                              == std::cv_status::no_timeout) {
                }
            }
        }
        // the first worker awake set the critical count
        if (!critical_count) {
            critical_count = ready_workers_.size();
            std::sort(ready_workers_.begin(), ready_workers_.end());
        }
        auto result = ready_workers_;
        critical_count--;
        if (should_stop)
            avail_workers_.erase(rank);
        // if being the last thread, clear the state
        if (critical_count == 0) {
            if (do_graph_sync_) {
                DisjointSetReset();
                unlockSyncStage();
                need_graph_sync_ = false;
            }
            size_t accum = accumulated_updates_ + ready_workers_.size();
            if (accum / ssp_bound_ > accumulated_updates_ / ssp_bound_)
                need_graph_sync_ = true;
            if (accum / sync_every_ > accumulated_updates_ / sync_every_) {
                stop_count_ = max_workers_;
                stopped_workers_.clear();
            }
            accumulated_updates_ = accum;
            ready_workers_.clear();
            cv_.notify_all();
        }
        return std::make_pair(result, should_stop);
    }

    std::pair<std::vector<int>, bool>
    getPartnerPairWise(int rank, int vertical_rank, size_t batch_id, float wait_time) {
        std::unique_lock<std::mutex> lock(mtx_);
        // must wait until the previous partial reduce decision finish
        while (critical_count)
            cv_.wait(lock);

        ready_workers_.push_back(rank);
        if (key_ == 0 && stop_count_ > 0 && !stopped_workers_.count(rank)) {
            // when stopped, set the condition so that all worker in the pipe
            // can see the condition and stop at the same batch id
            stop_count_--;
            stopped_workers_.insert(rank);
            setStopCondition(vertical_rank, batch_id);
        }
        const bool should_stop = checkStopCondition(vertical_rank, batch_id);
        avail_workers_.insert(rank);
        if (ready_workers_.size() == 1) {
            do_graph_sync_ = checkSyncType(key_, need_graph_sync_);
            if (avail_workers_.size() < max_workers_)
                do_graph_sync_ = false;
            // the first worker should set the wait time for others
            wake_time_ =
                system_clock::now() + microseconds(int(wait_time * 1000));
        }
        if (do_graph_sync_) {
            if (DisjointSetCanBeMergedBy(ready_workers_)) {
                cv_.notify_all();
            } else {
                while (!DisjointSetCanBeMergedBy(ready_workers_))
                    cv_.wait(lock);
            }
        } else {
            if (ready_workers_.size() == max_workers_) {
                // if worker number is enough, notify all
                cv_.notify_all();
            } else {
                while (ready_workers_.size() < max_workers_
                       && cv_.wait_until(lock, wake_time_)
                              == std::cv_status::no_timeout) {
                }
            }
        }
        // the first worker awake set the critical count
        if (!critical_count) {
            critical_count = ready_workers_.size();
            std::sort(ready_workers_.begin(), ready_workers_.end());
        }
        std::vector<int> result = ready_workers_;
        if (!do_graph_sync_) {
            for (int i = 0; i < ready_workers_.size(); i++) {
                if (ready_workers_[i] == rank) {
                    if (i % 2 == 1)
                        result = {ready_workers_[i-1], rank};
                    else if (i + 1 < ready_workers_.size())
                        result = {rank, ready_workers_[i+1]};
                    else
                        result = {rank};
                    break;
                }
            }
        }
        critical_count--;
        if (should_stop)
            avail_workers_.erase(rank);
        // if being the last thread, clear the state
        if (critical_count == 0) {
            if (do_graph_sync_) {
                DisjointSetReset();
                unlockSyncStage();
                need_graph_sync_ = false;
            }
            size_t accum = accumulated_updates_ + ready_workers_.size();
            if (accum / ssp_bound_ > accumulated_updates_ / ssp_bound_)
                need_graph_sync_ = true;
            if (accum / sync_every_ > accumulated_updates_ / sync_every_) {
                stop_count_ = max_workers_;
                stopped_workers_.clear();
            }
            accumulated_updates_ = accum;
            ready_workers_.clear();
            cv_.notify_all();
        }
        return std::make_pair(result, should_stop);
    }


    // register worker for initialization
    void registerWorker(int rank) {
        std::unique_lock<std::mutex> lock(mtx_);
        map_.emplace(rank, rank);
        if (map_.size() == max_workers_) {
            cv_.notify_all();
        } else {
            while (map_.size() < max_workers_)
                cv_.wait(lock);
        }
    }

    int critical_count = 0; // stop new worker from coming in, when the previous
                            // schedule is finishing

    inline size_t getMaxWorkers() { return max_workers_; }
private:
    // checkSyncType make sure only one key can do graph sync
    static bool checkSyncType(int key, bool graph_sync) {
        std::unique_lock<std::mutex> lock(mtx_static_);
        if (graph_sync) {
            if (graph_sync_stage_ == -1)
                graph_sync_stage_ = key;
            return graph_sync_stage_ == key;
        }
        return false;
    }
    static void unlockSyncStage() {
        std::unique_lock<std::mutex> lock(mtx_static_);
        graph_sync_stage_ = -1;
    }
    static bool checkStopCondition(int vertical_rank, size_t batch_id) {
        std::unique_lock<std::mutex> lock(mtx_static_);
        if (global_sync_map_.count(vertical_rank) == 0)
            return false;
        return global_sync_map_[vertical_rank] == batch_id;
    }
    static void setStopCondition(int vertical_rank, size_t batch_id) {
        std::unique_lock<std::mutex> lock(mtx_static_);
        global_sync_map_[vertical_rank] = batch_id;
    }
    //-----------------------Implement disjoint set---------------------------
    void DisjointSetReset() {
        for (auto &p : map_) {
            p.second = p.first;
        }
    }
    bool DisjointSetIsFullyMerged() {
        int head = DisjointSetFind(map_.begin()->first);
        for (const auto &p : map_) {
            if (DisjointSetFind(p.first) != head)
                return false;
        }
        return true;
    }
    bool DisjointSetCanBeMergedBy(std::vector<int> x) {
        std::unordered_map<int, bool> covered;
        for (const auto &p : map_)
            covered[DisjointSetFind(p.first)] = false;
        for (const auto &node : x)
            covered[DisjointSetFind(node)] = true;
        for (const auto &p : covered)
            if (!p.second) return false;
        return true;
    }
    int DisjointSetFind(int x) {
        return x == map_[x] ? x : (map_[x] = DisjointSetFind(map_[x]));
    }
    inline void DisjointSetMerge(int x, int y) {
        map_[DisjointSetFind(x)] = DisjointSetFind(y);
    }
    int key_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<int> ready_workers_;
    std::unordered_map<int, int> map_; // disjoint set
    system_clock::time_point wake_time_;
    const size_t max_workers_, ssp_bound_, sync_every_;
    size_t accumulated_updates_ = 0;
    bool need_graph_sync_ = false;
    bool do_graph_sync_ = false;
    std::unordered_set<int> stopped_workers_, avail_workers_;
    int stop_count_ = 0;

    static std::mutex mtx_static_;
    static int graph_sync_stage_;
    static std::unordered_map<int, size_t> global_sync_map_;
};

std::mutex PSHandler<PsfGroup::kPReduceScheduler>::ReduceStat::mtx_static_;
int PSHandler<PsfGroup::kPReduceScheduler>::ReduceStat::graph_sync_stage_ = -1;
std::unordered_map<int, size_t>
    PSHandler<PsfGroup::kPReduceScheduler>::ReduceStat::global_sync_map_;

PSHandler<PsfGroup::kPReduceScheduler>::~PSHandler() = default;

PSHandler<PsfGroup::kPReduceScheduler>::PSHandler() {
}

void PSHandler<PsfGroup::kPReduceScheduler>::serve(
    const PSFData<kPReduceInit>::Request &request,
    PSFData<kPReduceInit>::Response &response) {
    Key k = get<0>(request);
    int rank = get<1>(request);
    size_t max_workers = get<2>(request);
    size_t ssp_bound = get<3>(request);
    size_t sync_every = get<4>(request);
    map_mtx_.lock();
    if (!map_.count(k))
        map_.emplace(k, std::unique_ptr<ReduceStat>(new ReduceStat(
                            k, max_workers, ssp_bound, sync_every)));
    std::unique_ptr<ReduceStat> &obj = map_[k];
    map_mtx_.unlock();
    obj->registerWorker(rank);
    return;
}

void PSHandler<PsfGroup::kPReduceScheduler>::serve(
    const PSFData<kPReduceGetPartner>::Request &request,
    PSFData<kPReduceGetPartner>::Response &response) {
    Key k = get<0>(request);
    int rank = get<1>(request);
    int vertical_rank = get<2>(request);
    size_t batch_id = get<3>(request);
    float wait_time = get<4>(request);

    // get the reducestat
    map_mtx_.lock();
    CHECK(map_.count(k));
    std::unique_ptr<ReduceStat> &obj = map_[k];
    map_mtx_.unlock();
    std::pair<std::vector<int>, bool> result;
    if (obj->getMaxWorkers() > 8) {
        result = obj->getPartnerPairWise(rank, vertical_rank, batch_id, wait_time);
    } else {
        result = obj->getPartner(rank, vertical_rank, batch_id, wait_time);
    }
    // write return value
    get<0>(response).CopyFrom(result.first.data(), result.first.size());
    get<1>(response) = static_cast<int>(result.second);
    return;
}

} // namespace ps
