#ifndef ROLLINGAVERAGEH
#define ROLLINGAVERAGEH

#include <cassert>


template <typename T>
class RollingAverage
{
public:
    RollingAverage(std::size_t size) : m_size(size)
    {
        assert(m_size > 0);
        m_buf = new T[m_size]{};
    }

    ~RollingAverage(void)
    {
        delete[] m_buf;
        m_buf = nullptr;
    }

private:
    const std::size_t m_size;
    std::size_t m_saturated = 0, m_ptr = 0;
    T* m_buf {};

public:
    std::size_t size(void) const { return m_size; }
    std::size_t saturated(void) const { return m_saturated; }

    void push(T el)
    {
        m_buf[m_ptr] = el;
        if (++m_ptr >= m_size) m_ptr = 0;
        if (m_saturated < m_size) m_saturated++;
    }

    void clear(T el)
    {
        for (std::size_t i = 0; i < m_size; i++)
            m_buf[i] = el;
        m_ptr = 0;
        m_saturated = 0;
    }

    void clear(void)
    {
        for (std::size_t i = 0; i < m_size; i++)
            m_buf[i] = T{ 0 };
        m_ptr = 0;
        m_saturated = 0;
    }

    T sum(void) const
    {
        T acc {0};
        for (std::size_t i = 0; i < m_saturated; i++)
            acc += m_buf[i];
        return acc;
    }

    T avg(void) const
    {
        if (m_saturated == 0) return 0;
        return sum() / m_saturated;
    }
};

#endif // ROLLINGAVERAGEH
