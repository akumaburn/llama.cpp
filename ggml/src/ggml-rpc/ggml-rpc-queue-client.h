#ifndef GGML_RPC_QUEUE_CLIENT_H
#define GGML_RPC_QUEUE_CLIENT_H
#define RPC_QUEUE_ClIENT
#ifdef RPC_QUEUE_ClIENT
#define RPC_QUEUE
#include <unordered_map>
#include "./ggml-rpc-queue.h"
#include "./rpc_cmd.h"
#include "./socket.h"
struct rpc_client_task_t {
  void* output;
  size_t output_size;
  const void* input;
  size_t input_size;
  rpc_cmd cmd_byte;
};

struct rpc_client_queue_t : rpc_queue_t<rpc_client_task_t> {
  std::shared_ptr<socket_t> sockfd;
};

void process_client_queue(std::shared_ptr<rpc_client_queue_t>);

extern std::unordered_map<int, std::shared_ptr<rpc_client_queue_t>> queue_map;
#endif

#endif //GGML_RPC_QUEUE_CLIENT_H
