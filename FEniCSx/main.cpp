#include "poisson.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/io/VTKFile.h>
#include <dolfinx/la/petsc.h>
#include <utility>
#include <vector>
#include <iostream>
#include <span>

using namespace dolfinx;
using T = PetscScalar;
using U = double;

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  {
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::none);
    auto mesh_obj = mesh::create_rectangle<U>(
        MPI_COMM_WORLD,
        {{{-1.0, -1.0}, {1.0, 1.0}}},
        {80, 80},
        mesh::CellType::triangle,
        part);

    auto mesh = std::make_shared<mesh::Mesh<U>>(std::move(mesh_obj));
    mesh->topology_mutable()->create_connectivity(1, 2);

    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace<U>(functionspace_form_poisson_a, "u", mesh));

    auto kappa = std::make_shared<fem::Constant<T>>(1.0);

    auto f = std::make_shared<fem::Function<T>>(V);
    f->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> values(x.extent(1));
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            double r2_left = (x(0, p) + 0.5)*(x(0, p) + 0.5) + x(1, p)*x(1, p);
            double r2_right = (x(0, p) - 0.5)*(x(0, p) - 0.5) + x(1, p)*x(1, p);
            values[p] = 30.0 * (std::exp(-r2_left / 0.2) - std::exp(-r2_right / 0.2));
          }
          return {std::move(values), {values.size()}};
        });

    auto g = std::make_shared<fem::Function<T>>(V);
    g->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> values(x.extent(1), 0.0);
          return {std::move(values), {values.size()}};
        });

    auto a = std::make_shared<fem::Form<T>>(
        fem::create_form<T, U>(*form_poisson_a, {V, V}, {}, {{"kappa", kappa}}, {}, mesh));
    auto L = std::make_shared<fem::Form<T>>(
        fem::create_form<T, U>(*form_poisson_L, {V}, {{"f", f}, {"g", g}}, {}, {}, mesh));

    auto bdofs_left = fem::locate_dofs_geometrical(
        *V, [](auto x) { std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
            if (std::abs(x(0, p) + 1.0) < 1e-6) marker[p] = true;
          return marker; });

    auto bdofs_right = fem::locate_dofs_geometrical(
        *V, [](auto x) { std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
            if (std::abs(x(0, p) - 1.0) < 1e-6) marker[p] = true;
          return marker; });

    auto bdofs_top = fem::locate_dofs_geometrical(
        *V, [](auto x) { std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
            if (std::abs(x(1, p) - 1.0) < 1e-6) marker[p] = true;
          return marker; });

    auto bdofs_bottom = fem::locate_dofs_geometrical(
        *V, [](auto x) { std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
            if (std::abs(x(1, p) + 1.0) < 1e-6) marker[p] = true;
          return marker; });

    auto bc_left = std::make_shared<fem::DirichletBC<T>>(T(0), bdofs_left, V);
    auto bc_right = std::make_shared<fem::DirichletBC<T>>(T(1), bdofs_right, V);
    auto bc_top = std::make_shared<fem::DirichletBC<T>>(T(0), bdofs_top, V);
    auto bc_bottom = std::make_shared<fem::DirichletBC<T>>(T(0), bdofs_bottom, V);

    std::vector<std::shared_ptr<const fem::DirichletBC<T>>> bcs =
        {bc_left, bc_right, bc_top, bc_bottom};

    auto u = std::make_shared<fem::Function<T>>(V);

    Mat A_mat = fem::petsc::create_matrix(*a);
    la::petsc::Matrix A(A_mat, false);

    la::Vector<T> b(L->function_spaces()[0]->dofmap()->index_map,
                    L->function_spaces()[0]->dofmap()->index_map_bs());


    MatZeroEntries(A.mat());
    fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES),
                         *a, bcs);
    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);


    std::vector<std::int32_t> all_bdofs;
    all_bdofs.insert(all_bdofs.end(), bdofs_left.begin(), bdofs_left.end());
    all_bdofs.insert(all_bdofs.end(), bdofs_right.begin(), bdofs_right.end());
    all_bdofs.insert(all_bdofs.end(), bdofs_top.begin(), bdofs_top.end());
    all_bdofs.insert(all_bdofs.end(), bdofs_bottom.begin(), bdofs_bottom.end());

    std::sort(all_bdofs.begin(), all_bdofs.end());
    all_bdofs.erase(std::unique(all_bdofs.begin(), all_bdofs.end()), all_bdofs.end());

    fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A.mat(), INSERT_VALUES),
                         std::span(all_bdofs.data(), all_bdofs.size()), T(1));

    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

    b.set(0.0);
    fem::assemble_vector(b.mutable_array(), *L);

    for (auto& bc : bcs)
        bc->set(b.mutable_array());

    la::petsc::KrylovSolver solver(MPI_COMM_WORLD);
    la::petsc::options::set("ksp_type", "preonly");
    la::petsc::options::set("pc_type", "lu");
    solver.set_from_options();

    solver.set_operator(A.mat());
    la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
    la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);
    solver.solve(_u.vec(), _b.vec());

    io::VTKFile file(mesh->comm(), "u_square.pvd", "w");
    std::vector<std::reference_wrapper<const fem::Function<T>>> u_vec;
    u_vec.push_back(std::cref(*u));
    file.write(u_vec, 0.0);

  }

  PetscFinalize();
  return 0;
}
