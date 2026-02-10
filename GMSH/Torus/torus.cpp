#include <gmsh.h>
#include <iostream>
#include <set>
#include <vector>

int main(int argc, char* argv[]) {
  gmsh::initialize(argc, argv);

  gmsh::model::add("torus");

  double R = 3.0;
  double r_outer = 1.0;
  double r_inner = 0.6;

  int outer_torus = gmsh::model::occ::addTorus(0, 0, 0, R, r_outer);
  int inner_torus = gmsh::model::occ::addTorus(0, 0, 0, R, r_inner);

  std::vector<std::pair<int, int>> torus;
  std::vector<std::vector<std::pair<int, int>>> outDimTags;

  gmsh::model::occ::cut({{3, outer_torus}}, {{3, inner_torus}}, torus, outDimTags, 10);

  gmsh::model::occ::synchronize();

  gmsh::option::setNumber("Mesh.Algorithm3D", 4);
  gmsh::option::setNumber("Mesh.MeshSizeMin", 0.25);
  gmsh::option::setNumber("Mesh.MeshSizeMax", 0.25);


  gmsh::model::mesh::generate(3);
  gmsh::write("torus.msh");

  std::set<std::string> args(argv, argv + argc);
  if(!args.count("-nopopup")) {
    gmsh::fltk::run();
  }

  gmsh::finalize();
  return 0;
}
