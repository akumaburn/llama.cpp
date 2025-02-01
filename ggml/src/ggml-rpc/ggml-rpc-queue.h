#ifndef GGML_RPC_QUEUE_H
#define GGML_RPC_QUEUE_H
#ifdef RPC_QUEUE
#include <condition_variable>
#include <queue>
#include <thread>

template <typename T>
struct rpc_queue_t {
    std::queue<T> tasks;
    std::mutex mutex;
    std::condition_variable cond;
    volatile bool running;
};
#endif
#endif //GGML_RPC_QUEUE_H
