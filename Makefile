default: all

all:
	g++ -I${HOME}/softs/FreeImage/include modif_img.cpp -L${HOME}/softs/FreeImage/lib/ -lfreeimage -o modif_img.exe
	nvcc -I${HOME}/softs/FreeImage/include modif_img.cu -L${HOME}/softs/FreeImage/lib/ -lfreeimage -o test.exe
	nvcc -I${HOME}/softs/FreeImage/include modif_img.cu -L${HOME}/softs/FreeImage/lib/ -lfreeimage -o result.exe
clean:
	rm -f *.o modif_img.exe test.exe will.exe
