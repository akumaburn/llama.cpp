#include "./ggml-rpc-queue-client.h"
#ifdef RPC_QUEUE_ClIENT
typedef int sockfd_t;
bool send_data(sockfd_t sockfd, const void * data, size_t size);
bool recv_data(sockfd_t sockfd, void * data, size_t size);
void process_client_queue(std::shared_ptr<rpc_client_queue_t> _queue) {
    auto queue = _queue.get();
    while (queue->running) {
        std::unique_lock<std::mutex> lock(queue->mutex);
        queue->cond.wait(lock, [queue] { return !queue->tasks.empty() || !queue->running; });
        if (queue->tasks.empty()) {
            break;
        }

        rpc_client_task_t &task = queue->tasks.front();
        queue->tasks.pop();
        lock.unlock();
        auto sock = queue->sockfd->fd;
        if (!send_data(sock, &task.cmd_byte, sizeof(task.cmd_byte))) {
            queue->running = false;
            break;
        }
        if (!send_data(sock, &task.input_size, sizeof(task.input_size))) {
            queue->running = false;
            break;
        }
        if (!send_data(sock, task.input, task.input_size)) {
            queue->running = false;
            break;
        }
        // TODO: currently the output_size is always known, do we need support for commands with variable output size?
        // even if we do, we can skip sending output_size from the server for commands with known output size
        uint64_t out_size;
        if (!recv_data(sock, &out_size, sizeof(out_size))) {
            queue->running = false;
            break;
        }
        if (out_size != task.output_size) {
            queue->running = false;
            break;
        }
        if (!recv_data(sock, task.output, task.output_size)) {
            queue->running = false;
            break;
        }
    }
}
#endif