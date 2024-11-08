.PHONY: clean
test: ints_and_arrays.h ints_and_arrays.c ints_and_arrays_tests.c
	clang -Wall -o test ints_and_arrays.c ints_and_arrays_tests.c -lcriterion -lm
clean:
	rm -rf a.out test