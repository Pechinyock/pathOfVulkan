#include <iostream>
#define STB_IMAGE_IMPLEMENTATION 
#define TINYOBJLOADER_IMPLEMENTATION

#include "src/vulkan/triangleApp.cpp"

int main() {
	TriangleApp app;
	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cout << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}