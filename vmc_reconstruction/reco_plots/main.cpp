#include <xerus.h>
#include <xerus/misc/internal.h>
#include <xerus/misc/fileUtilities.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <fstream>

#include "json.hpp"
namespace nl = nlohmann;


std::set<size_t> sample_set(const size_t _setSize, const size_t _numSamples, const std::set<size_t>& _excluded = std::set<size_t>()) {
    REQUIRE(_setSize >= _numSamples+_excluded.size(), "NumSamples larger than the setSize.");
    std::set<size_t> sampleSet;
    std::uniform_int_distribution<size_t> dist(0, _setSize-1);

    while(sampleSet.size() < _numSamples) {
        const size_t candidate = dist(xerus::misc::randomEngine);
        if(!xerus::misc::contains(_excluded, candidate)) {
            sampleSet.insert(candidate);
        }
    }

    return sampleSet;
}

struct Sample {
    std::vector<double> params;
    xerus::Tensor solution;

    Sample() = default;

    Sample(const std::vector<double>& _params, const xerus::Tensor& _solution) : params(_params), solution(_solution) { }

    Sample(const std::string& _filename, const size_t _spacialDim, const size_t _M) {
        const auto content = xerus::misc::explode(xerus::misc::read_file(_filename), '\n');
        const auto paramStrings = xerus::misc::explode(content[0], ' ');
        REQUIRE(_M == paramStrings.size(), "Inconsitend stochastic dimension: " << _M << " vs " << paramStrings.size() << " in " << _filename);
        const auto valueStrings = xerus::misc::explode(content[1], ' ');
        REQUIRE( _spacialDim == valueStrings.size(), "Inconsitend spacial dimension: " << _spacialDim << " vs " << valueStrings.size() << " in " << _filename);

        for(const auto& par : paramStrings) { params.emplace_back(xerus::misc::from_string<double>(par)); }

        solution = xerus::Tensor({_spacialDim}, xerus::Tensor::Representation::Dense);
        for(size_t i = 0; i < _spacialDim; ++i) {
            REQUIRE(valueStrings[i] != "nan", "Encountered NaN");
            solution[i]= xerus::misc::from_string<double>(valueStrings[i]);
        }
    }
};

xerus::Tensor load_reference(const std::string& _filename, const size_t _spacialDim) {
    xerus::Tensor reference({_spacialDim});

    const auto refereceStrings = xerus::misc::explode(xerus::misc::read_file(_filename), ' ');
    REQUIRE(refereceStrings.size() == _spacialDim, "Fail: " << refereceStrings.size() << " vs " << _spacialDim );

    for(size_t i = 0; i < _spacialDim; ++i) {
        REQUIRE(refereceStrings[i] != "nan", "Encountered NaN");
        reference[i]= xerus::misc::from_string<double>(refereceStrings[i]);
    }

    return reference;
}


int main(int argc, char* argv[]) {
    const std::string basePath("../data/");

    const size_t maxSamples = 200000;
    const size_t maxQMCSamples = 10000;

    // Get experiment name
    REQUIRE(argc == 2, "Invalid number of arguments");
    const std::string expName = argv[1];
    const std::string expPath = basePath+expName+"/";

    LOG(info, "Loading settings for experiment #" << expName << " from " << expPath);

    const auto jsonFile = xerus::misc::read_file(expPath+"info.json");
    const auto jsonInfo = nl::json::parse(jsonFile);

    const size_t spacialDim = jsonInfo.at("fem dofs");
    const size_t M = jsonInfo.at("expansion size");
    const xerus::uq::PolynomBasis basisType = (jsonInfo.at("distribution") == "N" || jsonInfo.at("distribution") == 1) ? xerus::uq::PolynomBasis::Hermite : xerus::uq::PolynomBasis::Legendre;

    const size_t polyDegree = basisType == xerus::uq::PolynomBasis::Hermite ? 5 : 13;

    LOG(info, "Spacial dimension: " << spacialDim << " | Expansion size: " << M << " | PolyBasisSize: " << polyDegree << " | Polynom Type: " << (basisType == xerus::uq::PolynomBasis::Hermite ? "Hermite" : "Legendre") << " (" << jsonInfo.at("distribution") << ")");

    const auto files = xerus::misc::get_files(expPath+"mcSamps/");

    LOG(info, "There are " << files.size() << " datafiles.");

    std::vector<Sample> samples; samples.reserve(files.size());
    for(const auto& file : files) {
        if(file.substr(file.size() - 4) != ".dat" || file.substr(0, 8) != "mcSamps-") {
            LOG(info, "Ignoring non datafile: " << file);
            continue;
        }

        samples.emplace_back(expPath+"mcSamps/"+file, spacialDim, M);

        if(samples.size() == maxSamples) {
            LOG(info, "Stopping after reading " << maxSamples << " samples");
            break;
        }
    }


    const auto qmcFiles = xerus::misc::get_files(expPath+"qmcSamps/");
    LOG(info, "There are " << qmcFiles.size() << " QMC datafiles.");

    std::set<size_t> qmcIds;
    std::vector<Sample> qmcSamples(maxQMCSamples);
    for(const auto& file : qmcFiles) {
        if(file.substr(file.size() - 4) != ".dat" || file.substr(0, 9) != "qmcSamps-") {
            LOG(info, "Ignoring non datafile: " << file);
            continue;
        }

        const size_t qmcId = xerus::misc::from_string<size_t>(xerus::misc::explode(xerus::misc::explode(file, '.')[0], '-')[1])-1;
        if(qmcId >= maxQMCSamples) { continue; }
        qmcIds.insert(qmcId);

        qmcSamples[qmcId] = Sample(expPath+"qmcSamps/"+file, spacialDim, M);
    }
    REQUIRE(qmcIds.size() == maxQMCSamples && *qmcIds.begin() == 0 && *qmcIds.rbegin() == maxQMCSamples-1, "Insufficient QMC samples: " << qmcIds.size() << "/" << maxQMCSamples);

    LOG(info, "Load reference...");
//  const xerus::Tensor referenceM1 = load_reference(expPath+"/mc/"+"1000000m1.dat", spacialDim);
//  const xerus::Tensor referenceM2 = load_reference(expPath+"/mc/"+"1000000m2.dat", spacialDim);
    xerus::Tensor qmcM1, qmcM2;
    if(xerus::misc::file_exists(expPath+"/qmc/"+"1024000m1.dat")) {
        LOG(info, "Got 1024000 QMC Solution.");
        qmcM1 = load_reference(expPath+"/qmc/"+"1024000m1.dat", spacialDim);
        qmcM2 = load_reference(expPath+"/qmc/"+"1024000m2.dat", spacialDim);
    } else if(xerus::misc::file_exists(expPath+"/qmc/"+"512000m1.dat")) {
        LOG(info, "Got 512000 QMC Solution.");
        qmcM1 = load_reference(expPath+"/qmc/"+"512000m1.dat", spacialDim);
        qmcM2 = load_reference(expPath+"/qmc/"+"512000m2.dat", spacialDim);
    } else {
        LOG(info, "Got 256000 QMC Solution.");
        qmcM1 = load_reference(expPath+"/qmc/"+"256000m1.dat", spacialDim);
        qmcM2 = load_reference(expPath+"/qmc/"+"256000m2.dat", spacialDim);
    }


//  const double referenceM1Norm = referenceM1.frob_norm();
//  const double referenceM2Norm = referenceM2.frob_norm();
    const double qmcM1Norm = qmcM1.frob_norm();
    const double qmcM2Norm = qmcM2.frob_norm();

    auto statFile = xerus::misc::open_file_truncate("../solutions/"+expName+"-stat.dat");

    const std::vector<size_t> starts({400, 200, 100, 300});
    for(const auto start : starts) {
        for(size_t N = start; N <= 10000; N += 400) {
            LOG(info, "Running reconstruction using " << N << " samples.");

            LOG(info, "Calculate stats for QMC samples...");
            xerus::Tensor qmcSampM1({spacialDim});
            xerus::Tensor qmcSampM2({spacialDim});
            for(size_t k = 0; k < N; ++k) {
                qmcSampM1 += qmcSamples[k].solution;
                qmcSampM2 += xerus::entrywise_product(qmcSamples[k].solution, qmcSamples[k].solution);
            }
            qmcSampM1 /= double(N);
            qmcSampM2 /= double(N);


            const auto sampleSet = sample_set(samples.size(), N);

            LOG(info, "Calculate stats for samples...");
            xerus::Tensor sampM1({spacialDim});
            xerus::Tensor sampM2({spacialDim});
            for(const auto k : sampleSet) {
                sampM1 += samples[k].solution;
                sampM2 += xerus::entrywise_product(samples[k].solution, samples[k].solution);
            }
            sampM1 /= double(N);
            sampM2 /= double(N);

            LOG(info, "Create measurment set...");
            xerus::uq::UQMeasurementSet measurments;
            for(const auto i : sampleSet) {
                measurments.add(samples[i].params, samples[i].solution);
            }

            LOG(info, "Run reconstruction...");
            std::vector<size_t> dimensions(M+1, polyDegree); dimensions[0] = spacialDim;
            const auto solution = uq_ra_adf(measurments, basisType, dimensions);

            LOG(info, "Calculate stats for solution...");
            const auto detMoments = xerus::uq::det_moments(solution, basisType);
            const auto m1 = std::get<0>(detMoments);
            const auto m2 = std::get<1>(detMoments);

            double testError = 0.0;
            if(N+1000 <= samples.size()) {
                const auto testSet = sample_set(samples.size(), 1000, sampleSet);
                for (const auto s : testSet) {
                    const auto samp = xerus::uq::evaluate(solution, samples[s].params, basisType);
                    testError += frob_norm(samp - samples[s].solution)/frob_norm(samples[s].solution);
                }
                testError /= double(testSet.size());
            }


            LOG(info, "Save results...");
            const std::string path("../solutions/"+expName+"/");
            xerus::misc::create_folder_for_file(path+"solution.tsv");
            xerus::misc::save_to_file(solution,     path+"solution-"+xerus::misc::to_string(N)+".tsv",xerus::misc::FileFormat::TSV);

            LOG(info, "Samples vs QMC:        " << xerus::frob_norm(sampM1-qmcM1)/qmcM1Norm << " | " << xerus::frob_norm(sampM2-qmcM2)/qmcM2Norm);
            LOG(info, "QMC Samples vs QMC:        " << xerus::frob_norm(qmcSampM1-qmcM1)/qmcM1Norm << " | " << xerus::frob_norm(qmcSampM2-qmcM2)/qmcM2Norm);
            LOG(info, "Solution vs QMC:       " << xerus::frob_norm(m1-qmcM1)/qmcM1Norm << " | " << xerus::frob_norm(m2-qmcM2)/qmcM2Norm);
            LOG(info, "TestError:       " << testError);

            statFile << N << ' '
            << xerus::frob_norm(sampM1-qmcM1)/qmcM1Norm << ' ' << xerus::frob_norm(sampM2-qmcM2)/qmcM2Norm << ' '
            << xerus::frob_norm(qmcSampM1-qmcM1)/qmcM1Norm << ' ' << xerus::frob_norm(qmcSampM2-qmcM2)/qmcM2Norm << ' '
            << xerus::frob_norm(m1-qmcM1)/qmcM1Norm << ' ' << xerus::frob_norm(m2-qmcM2)/qmcM2Norm << ' '
            << testError
            << std::endl;
        }
    }
}
