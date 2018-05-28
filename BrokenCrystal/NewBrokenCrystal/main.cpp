#include "GLUTCallBack.h"

int main(int argc, char** argv)
{
	// 384, 216
	Scene scene;
	return glutmain(argc, argv, 960, 540, "SCV Path Tracing Demo", &scene);
}
