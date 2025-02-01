message(STATUS "Using RPC backend")

ggml_add_backend_library(ggml-rpc
                         ggml-rpc.cpp ggml-rpc-queue-client.cpp ggml-rpc-queue-server.cpp ggml-rpc-queue.cpp

                        )

if (WIN32)
    target_link_libraries(ggml-rpc PRIVATE ws2_32)
endif()
