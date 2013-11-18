#pragma once

/* A self-contained tool to do circular cycles
 */

class GridRadiusTraversal {
public:
    GridRadiusTraversal(int width, int height)
        : Width_(width)
        , Height_(height)
    {
    }

    template <class Action>
    void TraverseCircle(int x0, int y0, int r, Action action) {
        for (int x = std::max(0, x0 - r); x <= std::min(Width_ - 1, x0 + r); ++x) {
            float const dy = sqrt(r * static_cast<float>(r) - (x - x0) * static_cast<float>(x - x0));
            for (int y = std::max(0, static_cast<int>(ceil(y0 - dy)));
                 y <= std::min(Height_ - 1, static_cast<int>(floor(y0 + dy)));
                 ++y)
            {
                action(x, y);
            }
        }
    }

    template <class Action>
    void TraverseRectangle(int x0, int y0, int r, Action action) {
        for (int x = std::max(0, x0 - r); x <= std::min(Width_ - 1, x0 + r); ++x) {
            for (int y = std::max(0, y0 - r); y <= std::min(Height_ - 1, y0 + r); ++y) {
                action(x, y);
            }
        }
    }

private:
    int const Width_;
    int const Height_;
};
