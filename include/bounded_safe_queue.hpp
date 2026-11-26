#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <utility>

namespace steel {

enum class OverflowPolicy {
    kBlock,       // Block producer until space is available or queue is closed
    kDropOldest   // Drop oldest queued item when full (prevents OOM & latency buildup)
};

/**
 * @brief Thread-safe bounded FIFO queue with explicit shutdown and backpressure policies.
 */
template <typename T>
class BoundedSafeQueue {//线程安全队列,包含互斥锁、适合生产者-消费者并发场景
public:
    explicit BoundedSafeQueue(size_t capacity,
                              OverflowPolicy policy = OverflowPolicy::kBlock)
        : capacity_(capacity), policy_(policy) {
        if (capacity_ == 0) {
            throw std::invalid_argument("BoundedSafeQueue capacity must be > 0");
        }
    }

    ~BoundedSafeQueue() {
        close();
    }

    BoundedSafeQueue(const BoundedSafeQueue&) = delete;
    BoundedSafeQueue& operator=(const BoundedSafeQueue&) = delete;
    BoundedSafeQueue(BoundedSafeQueue&&) = delete;
    BoundedSafeQueue& operator=(BoundedSafeQueue&&) = delete;

    /**
     * @brief Push an item into the queue.
     * @return true if pushed, false if the queue is closed (or closed while waiting).
     */
    bool push(T item) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (closed_) {
            return false;
        }

        if (policy_ == OverflowPolicy::kBlock) {
            cv_not_full_.wait(lock, [this]() {
                return closed_ || queue_.size() < capacity_;
            });
            if (closed_) {
                return false;
            }
        } else {
            if (queue_.size() >= capacity_) {
                queue_.pop();
                ++dropped_count_;
            }
        }

        queue_.push(std::move(item));
        lock.unlock();
        cv_not_empty_.notify_one();
        return true;
    }

    /**
     * @brief Pop the front item into out_item, blocking until available or closed+empty.
     * @return true if an item was popped, false if closed and drained.
     */
    bool pop(T& out_item) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_not_empty_.wait(lock, [this]() {
            return closed_ || !queue_.empty();
        });

        if (queue_.empty()) {
            return false;
        }

        out_item = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        cv_not_full_.notify_one();
        return true;
    }

    /**
     * @brief Close the queue and wake up all blocked producers and consumers.
     */
    void close() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (closed_) {
                return;
            }
            closed_ = true;
        }
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return queue_.size();
    }

    size_t capacity() const noexcept {
        return capacity_;
    }

    size_t dropped_count() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return dropped_count_;
    }

    bool is_closed() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return closed_;
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mtx_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    size_t capacity_;
    OverflowPolicy policy_;
    bool closed_ = false;
    size_t dropped_count_ = 0;
};

}  // namespace steel
