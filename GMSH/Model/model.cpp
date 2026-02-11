#include <gmsh.h>
#include <iostream>
#include <set>
#include <vector>
#include <cmath>

int main(int argc, char* argv[]) {
  gmsh::initialize(argc, argv);
  gmsh::option::setNumber("General.Terminal", 1);
  gmsh::option::setNumber("General.Verbosity", 3);

  gmsh::model::add("Hand");

  try {
    gmsh::merge("hand2.stl");
  } catch(...) {
    gmsh::logger::write("Could not load STL mesh: bye!");
    gmsh::finalize();
    return 0;
  }

  gmsh::option::setNumber("Geometry.OCCAutoFix", 1);
  gmsh::option::setNumber("Geometry.OCCFixDegenerated", 1);
  gmsh::option::setNumber("Geometry.OCCFixSmallEdges", 1);
  gmsh::option::setNumber("Geometry.OCCFixSmallFaces", 1);
  gmsh::option::setNumber("Geometry.OCCSewFaces", 1);
  gmsh::option::setNumber("Geometry.OCCMakeSolids", 1);
  gmsh::option::setNumber("Geometry.Tolerance", 1e-3);
  gmsh::option::setNumber("Geometry.ToleranceBoolean", 1e-2);
  gmsh::option::setNumber("Geometry.OCCScaling", 1);

  gmsh::model::occ::synchronize();

  double angle = 40;
  bool forceParametrizablePatches = false;
  bool includeBoundary = true;
  double curveAngle = 180;

  gmsh::model::mesh::classifySurfaces(angle * M_PI / 180., includeBoundary,
                                          forceParametrizablePatches,
                                          curveAngle * M_PI / 180.);

  std::vector<std::pair<int, int> > s;
  gmsh::model::getEntities(s, 2);
  std::vector<int> sl;
  for(auto surf : s) sl.push_back(surf.second);
  int l = gmsh::model::geo::addSurfaceLoop(sl);
  gmsh::model::geo::addVolume({l});

  gmsh::model::geo::synchronize();

  gmsh::option::setNumber("Mesh.Algorithm", 5);
  gmsh::option::setNumber("Mesh.Algorithm3D", 1);
  gmsh::option::setNumber("Mesh.Optimize", 1);
  gmsh::option::setNumber("Mesh.OptimizeNetgen", 1);


  int f = gmsh::model::mesh::field::add("MathEval");
  gmsh::model::mesh::field::setString(f, "F", "1.0");
  gmsh::model::mesh::field::setAsBackgroundMesh(f);

  gmsh::model::mesh::generate(2);
  gmsh::model::mesh::generate(3);

  gmsh::write("Hand.msh");

  std::set<std::string> args(argv, argv + argc);
  if(!args.count("-nopopup")) {
    gmsh::fltk::run();
  }

  gmsh::finalize();
  return 0;
}
