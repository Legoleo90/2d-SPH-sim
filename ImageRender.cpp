#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <windows.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <chrono>


extern "C" unsigned char* GenerateImage(int width, int height, int channels);
extern "C" void ClearCudaData();

int main() {
    //Create window matching to the current screen size
    if (!glfwInit())
        return -1;
   
    int width = GetSystemMetrics(SM_CXSCREEN);
    int height = GetSystemMetrics(SM_CYSCREEN);
    int channels = 4;


    GLFWwindow* window = glfwCreateWindow(width, height, "OpenGL Triangle", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (glewInit() != GLEW_OK)
        return -1;


    //Creates a quad covering the entire screen
    float vertex[] = {
        -1,-1,0,   0,0,
        1,-1,0,    1,0,
        1, 1,0,    1,1,
        -1, 1,0,   0,1
    };


    unsigned int tris[] = {
        0,1,2,
        0,2,3
    };


    //Copy vertex data to gpu
    GLuint vertexBufferObject;
    glGenBuffers(1, &vertexBufferObject);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex), vertex, GL_STATIC_DRAW);


    //Copy triangle data to gpu
    GLuint trisBufferObject;
    glGenBuffers(1, &trisBufferObject);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, trisBufferObject);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(tris), tris, GL_STATIC_DRAW);


    //Create texture object
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);


    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);


    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,GL_UNSIGNED_BYTE, nullptr);


    //Construct and compile glsl shaders required for mapping the texture to the quad
    const char* vertexShaderSource = R"(
        #version 330 core


        layout(location = 0) in vec3 position;
        layout(location = 1) in vec2 texCoord;


        out vec2 uv;


        void main()
        {
            gl_Position = vec4(position, 1.0);
            uv = texCoord;
        }
    )";
    const char* fragmentShaderSource = R"(
        #version 330 core
       
        in vec2 uv;
       
        out vec4 FragColor;
       
        uniform sampler2D screenTexture;
       
        void main()
        {
            FragColor = texture(screenTexture, uv);
        }
    )";
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);


    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);


    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);


    glLinkProgram(shaderProgram);


    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);


    //link texture to cuda image
    cudaGraphicsResource* cudaResource;
    cudaGraphicsGLRegisterImage(&cudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));


    glActiveTexture(GL_TEXTURE0);
    glUniform1i(glGetUniformLocation(shaderProgram, "screenTexture"), 0);

    //Run render loop
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        //Copy cuda image from cuda kernel to openGl
        glFinish();
        auto start = std::chrono::high_resolution_clock::now();
        cudaGraphicsMapResources(1, &cudaResource);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout<<end-start<<std::endl;
        cudaArray_t array;
        cudaGraphicsSubResourceGetMappedArray(&array, cudaResource, 0, 0);

        unsigned char* gpuImage = GenerateImage(width, height, channels);
        cudaMemcpy2DToArray(array, 0, 0, gpuImage, width * channels, width * channels, height, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cudaResource);

        //Render quad with texture
        glUseProgram(shaderProgram);
   
        glBindTexture(GL_TEXTURE_2D, texture);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);


        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    ClearCudaData();
    return 0;
}

