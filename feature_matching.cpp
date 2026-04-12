#include <stdint.h>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/cvstd.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace std::chrono;

static const string PATH_OBJETO = "/home/lab/Desktop/Taller1corte2/Data/box.png";
static const string PATH_SCENE  = "/home/lab/Desktop/Taller1corte2/Data/box_in_scene.png";

enum MatcherType { BF_L2, BF_HAMMING, FLANN_KD };

struct ComboResult {
    int    id;
    string nombre;
    int    kp_objeto;
    int    kp_scene;
    int    good_matches;
    int    inliers;
    double tiempo_ms;
    bool   ok;
};

vector<DMatch> loweRatioTest(const vector<vector<DMatch>>& knn, float ratio = 0.75f)
{
    vector<DMatch> buenos;
    buenos.reserve(knn.size());
    for (const auto& m : knn)
        if (m.size() == 2 && m[0].distance < ratio * m[1].distance)
            buenos.push_back(m[0]);
    return buenos;
}

int contarInliers(const vector<KeyPoint>& kp1,
                  const vector<KeyPoint>& kp2,
                  const vector<DMatch>&   buenos)
{
    if ((int)buenos.size() < 4) return 0;
    vector<Point2f> p1, p2;
    for (const auto& m : buenos) {
        p1.push_back(kp1[m.queryIdx].pt);
        p2.push_back(kp2[m.trainIdx].pt);
    }
    Mat mask;
    Mat H = findHomography(p1, p2, RANSAC, 5.0, mask);
    if (H.empty() || mask.empty()) return 0;
    return countNonZero(mask);
}

ComboResult ejecutarCombo(int            id,
                          const string&  nombre,
                          Ptr<Feature2D> detector,
                          Ptr<Feature2D> descriptor,
                          MatcherType    tipo_matcher,
                          const Mat&     img_obj,
                          const Mat&     img_scene)
{
    ComboResult res;
    res.id     = id;
    res.nombre = nombre;
    res.ok     = false;

    auto t0 = high_resolution_clock::now();

    vector<KeyPoint> kp_obj, kp_scene;
    detector->detect(img_obj,   kp_obj);
    detector->detect(img_scene, kp_scene);

    Mat desc_obj, desc_scene;
    descriptor->compute(img_obj,   kp_obj,   desc_obj);
    descriptor->compute(img_scene, kp_scene, desc_scene);

    auto t1 = high_resolution_clock::now();
    res.tiempo_ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
    res.kp_objeto = (int)kp_obj.size();
    res.kp_scene  = (int)kp_scene.size();

    if (desc_obj.empty() || desc_scene.empty() || kp_obj.empty() || kp_scene.empty()) {
        res.good_matches = res.inliers = 0;
        return res;
    }

    vector<vector<DMatch>> knn_matches;
    if (tipo_matcher == BF_HAMMING) {
        BFMatcher::create(NORM_HAMMING, false)->knnMatch(desc_obj, desc_scene, knn_matches, 2);
    } else if (tipo_matcher == BF_L2) {
        BFMatcher::create(NORM_L2, false)->knnMatch(desc_obj, desc_scene, knn_matches, 2);
    } else {
        Mat f_obj, f_scene;
        desc_obj.convertTo(f_obj,   CV_32F);
        desc_scene.convertTo(f_scene, CV_32F);
        DescriptorMatcher::create(DescriptorMatcher::FLANNBASED)
            ->knnMatch(f_obj, f_scene, knn_matches, 2);
    }

    vector<DMatch> buenos = loweRatioTest(knn_matches, 0.75f);
    res.good_matches = (int)buenos.size();
    res.inliers      = contarInliers(kp_obj, kp_scene, buenos);
    res.ok           = (res.inliers >= 4);

    return res;
}

void imprimirTabla(const vector<ComboResult>& resultados)
{
    const int W = 82;
    cout << "\n" << string(W, '=') << "\n";
    cout << "  50 COMBINACIONES - Taller 1 Segundo Corte\n";
    cout << string(W, '-') << "\n";
    cout << left
         << setw(4)  << "ID"
         << setw(22) << "Combinacion"
         << setw(7)  << "KP_obj"
         << setw(7)  << "KP_scn"
         << setw(8)  << "GoodMch"
         << setw(8)  << "Inliers"
         << setw(11) << "Tiempo(ms)"
         << setw(8)  << "Estado"
         << "\n" << string(W, '-') << "\n";

    for (const auto& r : resultados) {
        cout << left
             << setw(4)  << r.id
             << setw(22) << r.nombre
             << setw(7)  << r.kp_objeto
             << setw(7)  << r.kp_scene
             << setw(8)  << r.good_matches
             << setw(8)  << r.inliers
             << setw(11) << fixed << setprecision(2) << r.tiempo_ms
             << setw(8)  << (r.ok ? "[ OK ]" : "[FAILED]")
             << "\n";
    }
    cout << string(W, '=') << "\n";
}

void imprimirRanking(const vector<ComboResult>& todos)
{
    vector<ComboResult> exitosos;
    for (const auto& r : todos)
        if (r.ok) exitosos.push_back(r);

    if (exitosos.empty()) { cout << "\nNo hay combinaciones exitosas.\n"; return; }

    auto best_inl = *max_element(exitosos.begin(), exitosos.end(),
        [](const ComboResult& a, const ComboResult& b){ return a.inliers < b.inliers; });
    auto best_mch = *max_element(exitosos.begin(), exitosos.end(),
        [](const ComboResult& a, const ComboResult& b){ return a.good_matches < b.good_matches; });
    auto fastest  = *min_element(exitosos.begin(), exitosos.end(),
        [](const ComboResult& a, const ComboResult& b){ return a.tiempo_ms < b.tiempo_ms; });

    int ok_count   = (int)exitosos.size();
    int fail_count = (int)todos.size() - ok_count;

    cout << "\n  RANKING GLOBAL (" << todos.size() << " combinaciones)\n";
    cout << "  " << string(50, '-') << "\n";
    cout << "  Mejor inliers  : [" << best_inl.id << "] " << best_inl.nombre
         << " (" << best_inl.inliers << " inliers)\n";
    cout << "  Mejor matches  : [" << best_mch.id << "] " << best_mch.nombre
         << " (" << best_mch.good_matches << " matches)\n";
    cout << "  Mas rapido     : [" << fastest.id  << "] " << fastest.nombre
         << " (" << fixed << setprecision(2) << fastest.tiempo_ms << " ms)\n";
    cout << "\n  Total [ OK ]   : " << ok_count
         << "\n  Total [FAILED] : " << fail_count
         << "\n  Total          : " << todos.size() << "\n\n";
}

int main()
{
    Mat fto_objeto = imread(PATH_OBJETO, IMREAD_GRAYSCALE);
    Mat fto_scene  = imread(PATH_SCENE,  IMREAD_GRAYSCALE);

    if (fto_objeto.empty() || fto_scene.empty()) {
        cout << "[ERROR] No se pudo abrir una o ambas imagenes.\n"
             << "  objeto : " << PATH_OBJETO << "\n"
             << "  escena : " << PATH_SCENE  << "\n";
        return -1;
    }

    cout << "\n=== Taller 1 Segundo Corte - Vision por Computador ===\n";
    cout << "Objeto : " << PATH_OBJETO << " (" << fto_objeto.cols << "x" << fto_objeto.rows << ")\n";
    cout << "Escena : " << PATH_SCENE  << " (" << fto_scene.cols  << "x" << fto_scene.rows  << ")\n\n";

    // ----------------------------------------------------------
    //  Instanciar algoritmos
    // ----------------------------------------------------------
    Ptr<SIFT>  feat_sift    = SIFT::create(0, 3, 0.04, 10, 1.6);
    Ptr<SURF>  feat_surf    = SURF::create(50, 4, 3, false, false);
    Ptr<FastFeatureDetector>      feat_fast     = FastFeatureDetector::create(20, true);
    Ptr<BriefDescriptorExtractor> feat_brief    = BriefDescriptorExtractor::create(32);  // 256 bits
    Ptr<BriefDescriptorExtractor> feat_brief16  = BriefDescriptorExtractor::create(16);  // 128 bits
    Ptr<BriefDescriptorExtractor> feat_brief64  = BriefDescriptorExtractor::create(64);  // 512 bits
    Ptr<ORB>   feat_orb     = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    Ptr<BRISK> feat_brisk30 = BRISK::create(30, 3, 1.0f);
    Ptr<BRISK> feat_brisk15 = BRISK::create(15, 3, 1.0f);
    Ptr<FREAK> feat_freak   = FREAK::create(true, true, 22.0f, 4);

    // ----------------------------------------------------------
    //  50 combinaciones en un solo bloque
    // ----------------------------------------------------------
    struct Combo {
        int            id;
        string         nombre;
        Ptr<Feature2D> detector;
        Ptr<Feature2D> descriptor;
        MatcherType    matcher;
    };

    vector<Combo> combos = {
        // ---- SIFT detector (7) --------------------------------
        {  0, "SIFT+SIFT+BF",      feat_sift,    feat_sift,    BF_L2      },
        {  1, "SIFT+SIFT+FLANN",   feat_sift,    feat_sift,    FLANN_KD   },
        {  2, "SIFT+SURF+BF",      feat_sift,    feat_surf,    BF_L2      },
        {  3, "SIFT+SURF+FLANN",   feat_sift,    feat_surf,    FLANN_KD   },
        {  4, "SIFT+BRIEF+BF",     feat_sift,    feat_brief,   BF_HAMMING },
        {  5, "SIFT+FREAK+BF",     feat_sift,    feat_freak,   BF_HAMMING },
        {  6, "SIFT+BRISK+BF",     feat_sift,    feat_brisk30, BF_HAMMING },
        // ---- SURF detector (8) --------------------------------
        {  7, "SURF+SIFT+BF",      feat_surf,    feat_sift,    BF_L2      },
        {  8, "SURF+SIFT+FLANN",   feat_surf,    feat_sift,    FLANN_KD   },
        {  9, "SURF+SURF+BF",      feat_surf,    feat_surf,    BF_L2      },
        { 10, "SURF+SURF+FLANN",   feat_surf,    feat_surf,    FLANN_KD   },
        { 11, "SURF+ORB+BF",       feat_surf,    feat_orb,     BF_HAMMING },
        { 12, "SURF+BRIEF+BF",     feat_surf,    feat_brief,   BF_HAMMING },
        { 13, "SURF+FREAK+BF",     feat_surf,    feat_freak,   BF_HAMMING },
        { 14, "SURF+BRISK+BF",     feat_surf,    feat_brisk30, BF_HAMMING },
        // ---- ORB detector (8) ---------------------------------
        { 15, "ORB+SIFT+BF",       feat_orb,     feat_sift,    BF_L2      },
        { 16, "ORB+SIFT+FLANN",    feat_orb,     feat_sift,    FLANN_KD   },
        { 17, "ORB+SURF+BF",       feat_orb,     feat_surf,    BF_L2      },
        { 18, "ORB+SURF+FLANN",    feat_orb,     feat_surf,    FLANN_KD   },
        { 19, "ORB+ORB+BF",        feat_orb,     feat_orb,     BF_HAMMING },
        { 20, "ORB+BRIEF+BF",      feat_orb,     feat_brief,   BF_HAMMING },
        { 21, "ORB+FREAK+BF",      feat_orb,     feat_freak,   BF_HAMMING },
        { 22, "ORB+BRISK+BF",      feat_orb,     feat_brisk30, BF_HAMMING },
        // ---- FAST detector (4) --------------------------------
        { 23, "FAST+ORB+BF",       feat_fast,    feat_orb,     BF_HAMMING },
        { 24, "FAST+BRIEF+BF",     feat_fast,    feat_brief,   BF_HAMMING },
        { 25, "FAST+FREAK+BF",     feat_fast,    feat_freak,   BF_HAMMING },
        { 26, "FAST+BRISK+BF",     feat_fast,    feat_brisk30, BF_HAMMING },
        // ---- BRISK thresh=30 (8) ------------------------------
        { 27, "BRISK+SIFT+BF",     feat_brisk30, feat_sift,    BF_L2      },
        { 28, "BRISK+SIFT+FLANN",  feat_brisk30, feat_sift,    FLANN_KD   },
        { 29, "BRISK+SURF+BF",     feat_brisk30, feat_surf,    BF_L2      },
        { 30, "BRISK+SURF+FLANN",  feat_brisk30, feat_surf,    FLANN_KD   },
        { 31, "BRISK+ORB+BF",      feat_brisk30, feat_orb,     BF_HAMMING },
        { 32, "BRISK+BRIEF+BF",    feat_brisk30, feat_brief,   BF_HAMMING },
        { 33, "BRISK+FREAK+BF",    feat_brisk30, feat_freak,   BF_HAMMING },
        { 34, "BRISK+BRISK+BF",    feat_brisk30, feat_brisk30, BF_HAMMING },
        // ---- BRISK thresh=15 (5) ------------------------------
        { 35, "BRISK15+SIFT+BF",   feat_brisk15, feat_sift,    BF_L2      },
        { 36, "BRISK15+SIFT+FLANN",feat_brisk15, feat_sift,    FLANN_KD   },
        { 37, "BRISK15+SURF+BF",   feat_brisk15, feat_surf,    BF_L2      },
        { 38, "BRISK15+SURF+FLANN",feat_brisk15, feat_surf,    FLANN_KD   },
        { 39, "BRISK15+ORB+BF",    feat_brisk15, feat_orb,     BF_HAMMING },
        // ---- BRIEF 16 bytes / 128 bits (5) --------------------
        { 40, "SIFT+BRIEF16+BF",   feat_sift,    feat_brief16, BF_HAMMING },
        { 41, "SURF+BRIEF16+BF",   feat_surf,    feat_brief16, BF_HAMMING },
        { 42, "BRISK+BRIEF16+BF",  feat_brisk30, feat_brief16, BF_HAMMING },
        { 43, "FAST+BRIEF16+BF",   feat_fast,    feat_brief16, BF_HAMMING },
        { 44, "ORB+BRIEF16+BF",    feat_orb,     feat_brief16, BF_HAMMING },
        // ---- BRIEF 64 bytes / 512 bits (5) --------------------
        { 45, "SIFT+BRIEF64+BF",   feat_sift,    feat_brief64, BF_HAMMING },
        { 46, "SURF+BRIEF64+BF",   feat_surf,    feat_brief64, BF_HAMMING },
        { 47, "BRISK+BRIEF64+BF",  feat_brisk30, feat_brief64, BF_HAMMING },
        { 48, "FAST+BRIEF64+BF",   feat_fast,    feat_brief64, BF_HAMMING },
        { 49, "ORB+BRIEF64+BF",    feat_orb,     feat_brief64, BF_HAMMING },
    };

    // ----------------------------------------------------------
    //  Ejecutar todas
    // ----------------------------------------------------------
    vector<ComboResult> resultados;

    for (auto& c : combos) {
        cout << "[" << setw(2) << c.id << "] " << left << setw(22) << c.nombre << "... ";
        cout.flush();
        ComboResult r = ejecutarCombo(c.id, c.nombre, c.detector, c.descriptor,
                                      c.matcher, fto_objeto, fto_scene);
        resultados.push_back(r);
        if (r.ok)
            cout << "[ OK ]    kp=" << setw(5) << r.kp_objeto
                 << " good=" << setw(4) << r.good_matches
                 << " inliers=" << setw(4) << r.inliers
                 << " t=" << fixed << setprecision(2) << r.tiempo_ms << "ms\n";
        else
            cout << "[FAILED]  kp=" << setw(5) << r.kp_objeto
                 << " good=" << setw(4) << r.good_matches
                 << " inliers=" << setw(4) << r.inliers << "\n";
    }

    // ----------------------------------------------------------
    //  Tabla y ranking final
    // ----------------------------------------------------------
    imprimirTabla(resultados);
    imprimirRanking(resultados);

    cout << "Presiona ENTER para salir...\n";
    cin.get();
    return 0;
}
