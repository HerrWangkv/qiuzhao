main: main.cxx dummy.o
	g++ -o main main.cxx dummy.o
dummy.o: dummy.cxx
	g++ -c -o dummy.o dummy.cxx
clean: 
	rm -f *.o