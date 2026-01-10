#include <gtest/gtest.h>

#include "kmeans_tatami/kmeans_tatami.hpp"

#include <numeric>
#include <random>
#include <vector>
#include <algorithm>
#include <cstddef>

TEST(Matrix, Simple) {
    int NR = 20, NC = 50;
    std::vector<double> buffer(NR * NC);
    std::iota(buffer.begin(), buffer.end(), 0);

    auto dmat = std::make_shared<tatami::DenseMatrix<double, int, std::vector<double> > >(NR, NC, buffer, false);
    kmeans_tatami::Matrix<int, double, double, int> tmat(std::move(dmat));
    EXPECT_EQ(tmat.num_dimensions(), NR);
    EXPECT_EQ(tmat.num_observations(), NC);

    {
        std::vector<double> tmp(NR), tmp2(NR);
        auto work = tmat.new_extractor();
        for (int c = 0; c < NC; ++c) {
            auto it = buffer.begin() + static_cast<std::size_t>(c) * static_cast<std::size_t>(NR);
            std::copy_n(it, NR, tmp.begin());
            auto ptr = work->get_observation(c);
            std::copy_n(ptr, NR, tmp2.begin());
            EXPECT_EQ(tmp, tmp2);
        }
    }

    {
        std::vector<double> tmp(NR), tmp2(NR);
        auto work = tmat.new_extractor(5, 18);
        for (int c = 5; c < 18; ++c) {
            auto it = buffer.begin() + static_cast<std::size_t>(c) * static_cast<std::size_t>(NR);
            std::copy_n(it, NR, tmp.begin());
            auto ptr = work->get_observation();
            std::copy_n(ptr, NR, tmp2.begin());
            EXPECT_EQ(tmp, tmp2);
        }
    }

    {
        std::vector<double> tmp(NR), tmp2(NR);
        std::vector<int> predictions { 5, 1, 2, 19, 6, 7, 4, 1, 11 };
        auto work = tmat.new_extractor(predictions.data(), predictions.size());
        for (auto c : predictions) {
            auto it = buffer.begin() + static_cast<std::size_t>(c) * static_cast<std::size_t>(NR);
            std::copy_n(it, NR, tmp.begin());
            auto ptr = work->get_observation();
            std::copy_n(ptr, NR, tmp2.begin());
            EXPECT_EQ(tmp, tmp2);
        }
    }
}

TEST(Matrix, DifferentType) {
    int NR = 20, NC = 50;
    std::vector<float> buffer(NR * NC);
    std::iota(buffer.begin(), buffer.end(), 0);

    kmeans_tatami::Matrix<int, double, double, int> tmat(std::make_shared<tatami::DenseMatrix<double, int, std::vector<float> > >(NR, NC, buffer, false));
    EXPECT_EQ(tmat.num_dimensions(), NR);
    EXPECT_EQ(tmat.num_observations(), NC);

    {
        std::vector<double> tmp(NR), tmp2(NR);
        auto work = tmat.new_extractor();
        for (int c = 0; c < NC; ++c) {
            auto it = buffer.begin() + static_cast<std::size_t>(c) * static_cast<std::size_t>(NR);
            std::copy_n(it, NR, tmp.begin());
            auto ptr = work->get_observation(c);
            std::copy_n(ptr, NR, tmp2.begin());
            EXPECT_EQ(tmp, tmp2);
        }
    }

    {
        std::vector<double> tmp(NR), tmp2(NR);
        auto work = tmat.new_extractor(3, 12);
        for (int c = 3; c < 12; ++c) {
            auto it = buffer.begin() + static_cast<std::size_t>(c) * static_cast<std::size_t>(NR);
            std::copy_n(it, NR, tmp.begin());
            auto ptr = work->get_observation();
            std::copy_n(ptr, NR, tmp2.begin());
            EXPECT_EQ(tmp, tmp2);
        }
    }

    {
        std::vector<double> tmp(NR), tmp2(NR);
        std::vector<int> predictions { 9, 5, 18, 7, 15, 8, 12, 19, 1, 10 };
        auto work = tmat.new_extractor(predictions.data(), predictions.size());
        for (auto c : predictions) {
            auto it = buffer.begin() + static_cast<std::size_t>(c) * static_cast<std::size_t>(NR);
            std::copy_n(it, NR, tmp.begin());
            auto ptr = work->get_observation();
            std::copy_n(ptr, NR, tmp2.begin());
            EXPECT_EQ(tmp, tmp2);
        }
    }
}

TEST(Matrix, Transposed) {
    int NR = 20, NC = 50;
    std::vector<double> buffer(NR * NC);
    std::iota(buffer.begin(), buffer.end(), 0);

    auto dmat = std::make_shared<tatami::DenseMatrix<double, int, std::vector<double> > >(NR, NC, buffer, true);
    kmeans_tatami::Matrix<int, double, double, int> tmat(dmat);
    EXPECT_EQ(tmat.num_dimensions(), NR);
    EXPECT_EQ(tmat.num_observations(), NC);

    auto ext = dmat->dense_column();
    {
        std::vector<double> tmp(NR), tmp2(NR);
        auto work = tmat.new_extractor();
        for (int c = 0; c < NC; ++c) {
            ext->fetch(c, tmp.data());
            auto ptr = work->get_observation(c);
            std::copy_n(ptr, NR, tmp2.begin());
            EXPECT_EQ(tmp, tmp2);
        }
    }

    {
        std::vector<double> tmp(NR), tmp2(NR);
        auto work = tmat.new_extractor(7, 15);
        for (int c = 7; c < 15; ++c) {
            ext->fetch(c, tmp.data());
            auto ptr = work->get_observation();
            std::copy_n(ptr, NR, tmp2.begin());
            EXPECT_EQ(tmp, tmp2);
        }
    }

    {
        std::vector<double> tmp(NR), tmp2(NR);
        std::vector<int> predictions { 6, 8, 8, 18, 3, 4, 6, 15, 4, 3, 15 };
        auto work = tmat.new_extractor(predictions.data(), predictions.size());
        for (auto c : predictions) {
            ext->fetch(c, tmp.data());
            auto ptr = work->get_observation();
            std::copy_n(ptr, NR, tmp2.begin());
            EXPECT_EQ(tmp, tmp2);
        }
    }
}

TEST(Kmeans, Full) {
    int NR = 20, NC = 500;
    std::vector<double> buffer(NR * NC);
    std::mt19937_64 rng(1234567);
    for (auto& b : buffer) {
        b = aarand::standard_normal(rng).first; // whatever, just throw away the second one.
    }

    auto dmat = std::make_shared<tatami::DenseMatrix<double, int, std::vector<double> > >(NR, NC, buffer, false);
    kmeans_tatami::Matrix<int, double, double, int> tmat(std::move(dmat));
    kmeans::SimpleMatrix<int, double> smat(NR, NC, buffer.data());
    const int k = 10;

    // Random + Lloyd
    {
        auto tres = kmeans::compute(
            tmat,
            kmeans::InitializeRandom<int, double, int, double>(),
            kmeans::RefineLloyd<int, double, int, double>(),
            k
        );

        auto sres = kmeans::compute(
            smat,
            kmeans::InitializeRandom<int, double, int, double>(),
            kmeans::RefineLloyd<int, double, int, double>(),
            k
        );

        EXPECT_EQ(tres.clusters, sres.clusters);
        EXPECT_EQ(tres.centers, sres.centers);
        EXPECT_EQ(tres.details.sizes, sres.details.sizes);
        EXPECT_EQ(tres.details.iterations, sres.details.iterations);
    }

    // Kmeans++, Hartigan Wong.
    {
        auto tres = kmeans::compute(
            tmat,
            kmeans::InitializeKmeanspp<int, double, int, double>(),
            kmeans::RefineHartiganWong<int, double, int, double>(),
            k
        );

        auto sres = kmeans::compute(
            smat,
            kmeans::InitializeKmeanspp<int, double, int, double>(),
            kmeans::RefineHartiganWong<int, double, int, double>(),
            k
        );

        EXPECT_EQ(tres.clusters, sres.clusters);
        EXPECT_EQ(tres.centers, sres.centers);
        EXPECT_EQ(tres.details.sizes, sres.details.sizes);
        EXPECT_EQ(tres.details.iterations, sres.details.iterations);
    }

    // Variance partitioning, Minibatch
    {
        auto tres = kmeans::compute(
            tmat,
            kmeans::InitializeVariancePartition<int, double, int, double>(),
            kmeans::RefineMiniBatch<int, double, int, double>(),
            k
        );

        auto sres = kmeans::compute(
            smat,
            kmeans::InitializeVariancePartition<int, double, int, double>(),
            kmeans::RefineMiniBatch<int, double, int, double>(),
            k
        );

        EXPECT_EQ(tres.clusters, sres.clusters);
        EXPECT_EQ(tres.centers, sres.centers);
        EXPECT_EQ(tres.details.sizes, sres.details.sizes);
        EXPECT_EQ(tres.details.iterations, sres.details.iterations);
    }
}
