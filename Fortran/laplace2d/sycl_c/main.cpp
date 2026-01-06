extern "C" {
    void laplace_(); // trailing underscore depends on compiler name mangling
#if defined(__INTEL_CLANG_COMPILER)
    void for_rtl_init_(int *argc, char **argv);
    void for_rtl_finish_();
#endif
}

int main(int argc, char **argv) {

#if defined(__INTEL_CLANG_COMPILER)
    for_rtl_init_(&argc, argv);
#endif

// 	Name of subroutine program in F90 file
    laplace_();

#if defined(__INTEL_CLANG_COMPILER)
    for_rtl_finish_();
#endif

    return 0;
}
