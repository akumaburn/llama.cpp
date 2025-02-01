#ifndef SOCKET_H
#define SOCKET_H
#include "ggml-impl.h"
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#else
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif

#ifdef _WIN32
typedef SOCKET sockfd_t;
using ssize_t = __int64;
#else
typedef int sockfd_t;
#endif
struct socket_t {
    sockfd_t fd;
    socket_t(sockfd_t fd) : fd(fd) {}
    ~socket_t() {
        GGML_PRINT_DEBUG("[%s] closing socket %d\n", __func__, this->fd);
#ifdef _WIN32
        closesocket(this->fd);
#else
        close(this->fd);
#endif
    }
};
#endif //SOCKET_H
