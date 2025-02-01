#include "./ggml-rpc-queue-server.h"
#ifdef RPC_QUEUE_SERVER
#include <variant>
#include <vector>

#include "./rpc_cmd.h"
#include "./rpc_msg.h"
#include "./ggml-rpc-server.h"
#include "ggml-backend.h"
typedef int sockfd_t;
bool recv_msg(sockfd_t sockfd, void * msg, size_t msg_size);
bool send_msg(sockfd_t sockfd, const void * msg, size_t msg_size);
bool recv_data(sockfd_t sockfd, void * data, size_t size);
bool recv_msg(sockfd_t sockfd, std::vector<uint8_t> & input);
struct rpc_server_task_t {
    rpc_cmd cmd;
    typedef std::variant<rpc_msg_alloc_buffer_req, rpc_msg_get_alloc_size_req,
                    rpc_msg_buffer_get_base_req, rpc_msg_free_buffer_req,
                    rpc_msg_buffer_clear_req, std::vector<uint8_t>,
                    rpc_msg_get_tensor_req, rpc_msg_copy_tensor_req,
                    rpc_msg_init_tensor_req> req_t;
    req_t req;
    std::variant<rpc_msg_alloc_buffer_rsp, rpc_msg_get_alloc_size_rsp,
                    rpc_msg_get_alignment_rsp, rpc_msg_get_max_size_rsp,
                    rpc_msg_buffer_get_base_rsp, std::vector<uint8_t>,
                    rpc_msg_copy_tensor_rsp, rpc_msg_graph_compute_rsp,
                    rpc_msg_get_device_memory_rsp, bool> rsp;
    sockfd_t sockfd;
    std::mutex response_mutex;
    rpc_server_task_t(rpc_server_task_t&& t) : cmd(t.cmd), req(t.req), rsp(t.rsp), sockfd(t.sockfd) {}
    rpc_server_task_t(rpc_cmd cmd, req_t req, sockfd_t sockfd) : cmd(cmd), req(req), sockfd(sockfd) {}
};

struct rpc_server_worker_context {
    std::shared_ptr<rpc_queue_t<rpc_server_task_t>> queue;
    ggml_backend_t                                  backend;
    size_t                                          free_mem;
    size_t                                          total_mem;
};

void process_server_queue(rpc_server_worker_context * ctx);

bool send_response(const rpc_server_task_t& task);
bool send_response(const rpc_server_task_t & task) {
    size_t       response_size = 0;
    const void * response_data = nullptr;

    switch (task.cmd) {
        case rpc_cmd::RPC_CMD_ALLOC_BUFFER:
            response_data = &std::get<rpc_msg_alloc_buffer_rsp>(task.rsp);
            response_size = sizeof(rpc_msg_alloc_buffer_rsp);
            break;
        case RPC_CMD_GET_ALIGNMENT:
            response_data = &std::get<rpc_msg_get_alignment_rsp>(task.rsp);
            response_size = sizeof(rpc_msg_get_alignment_rsp);
            break;
        case RPC_CMD_GET_MAX_SIZE:
            response_data = &std::get<rpc_msg_get_max_size_rsp>(task.rsp);
            response_size = sizeof(rpc_msg_get_max_size_rsp);
            break;
        case RPC_CMD_BUFFER_GET_BASE:
            response_data = &std::get<rpc_msg_buffer_get_base_rsp>(task.rsp);
            response_size = sizeof(rpc_msg_buffer_get_base_rsp);
            break;
        case RPC_CMD_COPY_TENSOR:
            response_data = &std::get<rpc_msg_copy_tensor_rsp>(task.rsp);
            response_size = sizeof(rpc_msg_copy_tensor_rsp);
            break;
        case RPC_CMD_INIT_TENSOR:
        case RPC_CMD_SET_TENSOR:
            response_data = &std::get<bool>(task.rsp);
            response_size = 1;
            break;
        case RPC_CMD_GET_TENSOR:
            response_data = std::get<std::vector<uint8_t>>(task.rsp).data();
            response_size = std::get<std::vector<uint8_t>>(task.rsp).size();
            break;
        case RPC_CMD_GRAPH_COMPUTE:
            response_data = &std::get<rpc_msg_graph_compute_rsp>(task.rsp).result;
            response_size = sizeof(rpc_msg_graph_compute_rsp::result);
            break;
        case RPC_CMD_GET_DEVICE_MEMORY:
            response_data = &std::get<rpc_msg_get_device_memory_rsp>(task.rsp);
            response_size = sizeof(rpc_msg_get_device_memory_rsp);
            break;
        case RPC_CMD_BUFFER_CLEAR:
        case RPC_CMD_FREE_BUFFER:
            // No response data for this command
            response_size = 0;
            break;
        default:
            response_size = 0;
    }
    return send_msg(task.sockfd, response_data, response_size);
}

void process_server_queue(rpc_server_worker_context * ctx) {
    rpc_queue_t<rpc_server_task_t>* queue = ctx->queue.get();
    rpc_server server(ctx->backend);
    while (queue->running) {
        std::unique_lock<std::mutex> lock(queue->mutex);
        // ReSharper disable once CppDFAConstantConditions
        queue->cond.wait(lock, [queue] { return !queue->tasks.empty() || !queue->running; });

        if (queue->tasks.empty()) {
            break;
        }
        rpc_server_task_t &task = ctx->queue->tasks.front();
        queue->tasks.pop();
        lock.unlock();
        switch (task.cmd) {
            case RPC_CMD_ALLOC_BUFFER:
                server.alloc_buffer(std::get<rpc_msg_alloc_buffer_req>(task.req), std::get<rpc_msg_alloc_buffer_rsp>(task.rsp));
                std::get<bool>(task.rsp) = true;
                break;
            case RPC_CMD_GET_ALIGNMENT:
                server.get_alignment(std::get<rpc_msg_get_alignment_rsp>(task.rsp));
                std::get<bool>(task.rsp) = true;
                break;
            case RPC_CMD_GET_MAX_SIZE:
                server.get_max_size(std::get<rpc_msg_get_max_size_rsp>(task.rsp));
                std::get<bool>(task.rsp) = true;
                break;
            case RPC_CMD_BUFFER_GET_BASE:
                std::get<bool>(task.rsp) = server.buffer_get_base(std::get<rpc_msg_buffer_get_base_req>(task.req), std::get<rpc_msg_buffer_get_base_rsp>(task.rsp));
                break;
            case RPC_CMD_FREE_BUFFER:
                std::get<bool>(task.rsp) = server.free_buffer(std::get<rpc_msg_free_buffer_req>(task.req));
                break;
            case RPC_CMD_BUFFER_CLEAR:
                std::get<bool>(task.rsp)= server.buffer_clear(std::get<rpc_msg_buffer_clear_req>(task.req));
                break;
            case RPC_CMD_SET_TENSOR:
                std::get<bool>(task.rsp) = server.set_tensor(std::get<std::vector<uint8_t>>(task.req));
                break;
            case RPC_CMD_GET_TENSOR:
                std::get<bool>(task.rsp) = server.get_tensor(std::get<rpc_msg_get_tensor_req>(task.req), std::get<std::vector<uint8_t>>(task.rsp));
                break;
            case RPC_CMD_COPY_TENSOR:
                std::get<bool>(task.rsp) = server.copy_tensor(std::get<rpc_msg_copy_tensor_req>(task.req), std::get<rpc_msg_copy_tensor_rsp>(task.rsp));
                break;
            case RPC_CMD_GRAPH_COMPUTE:
                std::get<bool>(task.rsp) = server.graph_compute(std::get<std::vector<uint8_t>>(task.req), std::get<rpc_msg_graph_compute_rsp>(task.rsp));
                break;
            case RPC_CMD_GET_DEVICE_MEMORY:
                std::get<rpc_msg_get_device_memory_rsp>(task.rsp).free_mem  = ctx->free_mem;
                std::get<rpc_msg_get_device_memory_rsp>(task.rsp).total_mem = ctx->total_mem;
                std::get<bool>(task.rsp) = true;
                break;
            case RPC_CMD_INIT_TENSOR:
                std::get<bool>(task.rsp) = server.init_tensor(std::get<rpc_msg_init_tensor_req>(task.req));
                break;
            case RPC_CMD_GET_ALLOC_SIZE:
                std::get<bool>(task.rsp) = server.get_alloc_size(std::get<rpc_msg_get_alloc_size_req>(task.req), std::get<rpc_msg_get_alloc_size_rsp>(task.rsp));
                break;
            default:
                std::get<bool>(task.rsp) = false;
                break;
        }

        std::lock_guard<std::mutex> response_lock(task.response_mutex);
        if (!send_response(task)) {
            std::get<bool>(task.rsp) = true;
        }
    }
}

//if you change, then do synchronize change with such name function (ggml-rpc.cpp)
static void rpc_serve_client(ggml_backend_t backend, sockfd_t sockfd, std::shared_ptr<rpc_queue_t<rpc_server_task_t>> _queue) {
    auto queue = _queue.get();
    rpc_server server(backend);
    while (true) {
        rpc_cmd cmd;
        if (!recv_data(sockfd, &cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            fprintf(stderr, "Unknown command: %d\n", cmd);
            break;
        }
        rpc_server_task_t::req_t req;
        switch (cmd) {
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sockfd, &request, sizeof(request))) {
                    return;
                }
                req = request;
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sockfd, &request, sizeof(request))) {
                    return;
                }
                req = request;
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                if (!recv_msg(sockfd, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                if (!recv_msg(sockfd, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sockfd, &request, sizeof(request))) {
                    return;
                }
                req = request;
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sockfd, &request, sizeof(request))) {
                    return;
                }
                req = request;
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sockfd, &request, sizeof(request))) {
                    return;
                }
                req = request;
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                if (!recv_msg(sockfd, input)) {
                    return;
                }
                req = input;
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sockfd, &request,sizeof(request))) {
                    return;
                }
                req = request;
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sockfd, &request, sizeof(request))) {
                    return;
                }
                req = request;
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sockfd, &request, sizeof(request))) {
                    return;
                }
                req = request;
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sockfd, input)) {
                    return;
                }
                req = input;
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                if (!recv_msg(sockfd, nullptr, 0)) {
                    return;
                }
                break;
            }
            default: {
                fprintf(stderr, "Unknown command: %d\n", cmd);
                return;
            }
        }
        std::lock_guard<std::mutex> lock(queue->mutex);
        queue->tasks.push(rpc_server_task_t{cmd, req, sockfd});
        queue->cond.notify_one();
    }
}
#endif