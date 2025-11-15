
#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <utility>

using namespace sycl;

// ---------------------- GPUContext ----------------------
struct GPUContext {
    queue q;

    int width;
    int height;
    int numN;
    size_t totalCells;
    size_t totalDeltaElems; // totalCells * numN

    // device pointers (device-USM)
    int *res_dev = nullptr;       // resources per cell
    int *type_dev = nullptr;      // cell type per cell (if needed)
    int *delta_in_dev = nullptr;  // 3D incoming: totalCells * numN
    int *outgoing_dev = nullptr;  // 2D outgoing: totalCells
    std::pair<int,int> *neigh_dev = nullptr; // neighbor offsets

    GPUContext(queue q_, int w, int h, int n,
               const std::vector<int>& res_host,
               const std::vector<int>& type_host,
               const std::vector<std::pair<int,int>>& neigh_host)
      : q(q_), width(w), height(h), numN(n)
    {
        totalCells = size_t(width) * size_t(height);
        totalDeltaElems = totalCells * size_t(numN);

        // allocate device USM memory (device pointers for best perf)
        res_dev      = malloc_device<int>(totalCells, q);
        type_dev     = malloc_device<int>(totalCells, q);
        delta_in_dev = malloc_device<int>(totalDeltaElems, q);
        outgoing_dev = malloc_device<int>(totalCells, q);
        neigh_dev    = malloc_device<std::pair<int,int>>(numN, q);

        // copy initial host data
        q.memcpy(res_dev, res_host.data(), sizeof(int) * totalCells);
        q.memcpy(type_dev, type_host.data(), sizeof(int) * totalCells);
        q.memcpy(neigh_dev, neigh_host.data(), sizeof(std::pair<int,int>) * numN);

        q.wait();
    }

    ~GPUContext() {
        if (res_dev) free(res_dev, q);
        if (type_dev) free(type_dev, q);
        if (delta_in_dev) free(delta_in_dev, q);
        if (outgoing_dev) free(outgoing_dev, q);
        if (neigh_dev) free(neigh_dev, q);
    }
};

// ---------------------- small device helpers ----------------------
struct CoordHelper {
    int width, height;
    [[nodiscard]] inline int flat(int r, int q) const { return r * width + q; }
    [[nodiscard]] inline int maskInBounds(int r, int q) const {
        return (r >= 0) & (r < height) & (q >= 0) & (q < width);
    }
    [[nodiscard]] inline int clampR(int r) const {
        return r < 0 ? 0 : (r >= height ? height - 1 : r);
    }
    [[nodiscard]] inline int clampQ(int q) const {
        return q < 0 ? 0 : (q >= width ? width - 1 : q);
    }
};

struct DeltaIndex {
    int numN;
    [[nodiscard]] inline size_t idx(int flat, int n) const {
        return size_t(flat) * size_t(numN) + size_t(n);
    }
};

// ---------------------- Pass 1 functor (scatter: set incoming and outgoing) ----------------------
struct Pass1Scatter {
    int width, height, numN;
    int *res;               // device pointer
    int *type;              // optional
    int *delta_in;          // device pointer
    int *outgoing;          // device pointer
    const std::pair<int,int>* neigh; // device pointer

    // device helpers (small functions)
    inline int canSendMask(int resources, int typeVal) const {
        // resources > 0 and cell type is 'Cell' (assume 1)
        return (resources > 0) & (typeVal == 1);
    }
    inline int chooseNeighbor(int r, int q) const {
        // deterministic policy; replace with RNG if desired
        int v = (r + q) % numN;
        return v < 0 ? v + numN : v;
    }

    void operator()(sycl::id<2> id) const {
        const int r = int(id[0]);
        const int q_ = int(id[1]);

        const CoordHelper CH{width, height};
        const DeltaIndex DI{numN};

        const int flat = CH.flat(r, q_);

        // Read cell state
        const int resources = res[flat];
        const int typeVal = type[flat];

        // Can we send?
        const int canSend = canSendMask(resources, typeVal); // 0 or 1

        // pick one neighbor index
        const int chosen = chooseNeighbor(r, q_);
        const auto off = neigh[chosen];
        const int nr = r + int(off.second);
        const int nq = q_ + int(off.first);

        // validity mask for neighbor (branchless)
        const int validNeighbor = CH.maskInBounds(nr, nq);

        // final send mask
        const int sendMask = canSend & validNeighbor; // 0 or 1

        // write to incoming delta slot for chosen neighbor
        const size_t dst_in = DI.idx(flat, chosen);
        // each thread writes exclusively to its own slice + chosen neighbor index -> no atomic needed
        delta_in[dst_in] = sendMask;

        // write outgoing indicator for this cell (0/1)
        outgoing[flat] = sendMask;
    }
};

// ---------------------- Pass 2 functor (reduce incoming and apply outgoing) ----------------------
struct Pass2Apply {
    int width, height, numN;
    int *res;
    int *delta_in;
    int *outgoing;

    void operator()(item<2> item) const {
        const int r  = int(item.get_id(0));
        const int q_ = int(item.get_id(1));

        if (r >= height || q_ >= width) return;

        const int flat = r * width + q_;
        const int base = flat * numN;

        // sum incoming
        int gain = 0;
        for (int n = 0; n < numN; ++n)
            gain += delta_in[ base + n ];

        // outgoing is 0 or 1
        int loss = outgoing[flat];

        res[flat] += (gain - loss);
    }
};

// ---------------------- Simulator wrapper ----------------------
struct ResourcesSimulator {
    GPUContext ctx;

    ResourcesSimulator(queue q,
              int width, int height,
              const std::vector<int>& res_host,
              const std::vector<int>& type_host,
              const std::vector<std::pair<int,int>>& neigh_host)
    : ctx(q, width, height, int(neigh_host.size()), res_host, type_host, neigh_host)
    {}

    void step() {
        // 1) clear delta_in and outgoing on device (one fast device ops)
        ctx.q.fill<int>(ctx.delta_in_dev, 0, ctx.totalDeltaElems);
        ctx.q.fill<int>(ctx.outgoing_dev, 0, ctx.totalCells);

        // 2) Pass 1 scatter (parallel_for over exact domain range)
        {
            range<2> global(ctx.height, ctx.width);
            Pass1Scatter p1{ctx.width, ctx.height, ctx.numN,
                            ctx.res_dev, ctx.type_dev, ctx.delta_in_dev, ctx.outgoing_dev, ctx.neigh_dev};

            ctx.q.submit([&](handler& h) {
                h.parallel_for(global, p1);
            });
        }

        // 3) Pass 2 apply (parallel_for over exact domain range)
        {
            range<2> global(ctx.height, ctx.width);
            Pass2Apply p2{ctx.width, ctx.height, ctx.numN,
                          ctx.res_dev, ctx.delta_in_dev, ctx.outgoing_dev };

            ctx.q.submit([&](handler& h) {
                h.parallel_for(global, p2);
            });
        }

        // do not ctx.q.wait() here: caller chooses when to wait (render, gather stats)
    }

    // copy resources to host only when needed
    std::vector<int> copyResourcesToHost() {
        std::vector<int> out;
        out.resize(ctx.totalCells);
        ctx.q.memcpy(out.data(), ctx.res_dev, sizeof(int) * ctx.totalCells);
        ctx.q.wait();
        return out;
    }
};
