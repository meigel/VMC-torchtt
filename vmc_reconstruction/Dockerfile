FROM quay.io/fenicsproject/stable:2017.1.0
USER root
WORKDIR "/home/fenics/local"
RUN git clone https://github.com/jluttine/suitesparse.git
WORKDIR "/home/fenics/local/suitesparse" 
RUN make
RUN add-apt-repository "deb http://archive.canonical.com/ubuntu/ $(lsb_release -sc) partner" && \ 
	apt-get update && \
	apt-get install -y binutils-dev && \
	apt-get install -y libiberty-dev && \
	apt-get install -y liblapack* && \
	apt-get install -y libboost-all-dev
WORKDIR "/home/fenics/local"
RUN git clone https://git.hemio.de/xerus/xerus.git
WORKDIR "/home/fenics/local/xerus"
RUN git checkout development
RUN cp config.mk.default config.mk
RUN sed -i "s|# BUILD_PYTHON_BINDINGS = TRUE|BUILD_PYTHON_BINDINGS = TRUE |g" config.mk 
RUN sed -i "s|HIGH_OPTIMIZATION = TRUE|# HIGH_OPTIMIZATION = TRUE|g" config.mk 
RUN sed -i "s|# OTHER += -fopenmp|OTHER += -fopenmp|g" config.mk 
RUN sed -i "s|SUITESPARSE = -lcholmod -lspqr|SUITESPARSE = -L/home/fenics/local/suitesparse/lib -lcholmod -lspqr |g" config.mk 
RUN sed -i "s|# INSTALL_LIB_PATH = /usr/local/lib/         # Path where to install the libxerus.so shared library.|INSTALL_LIB_PATH = /usr/local/lib|g" config.mk 
RUN sed -i "s|# INSTALL_HEADER_PATH = /usr/local/include/  # Path where to install the xerus header files.|INSTALL_HEADER_PATH = /usr/local/include|g" config.mk 
RUN sed -i "s|# INSTALL_PYTHON2_PATH = /usr/local/lib/python2.7/site-packages/    # Path for the installation of the python2 bindings.|INSTALL_PYTHON2_PATH = /usr/local/lib/python2.7/site-packages|g" config.mk 
RUN sed -i "s|# OTHER += -I /path/to/include|OTHER += -I /home/fenics/local/suitesparse/include |g" config.mk
RUN ln -s /home/fenics/local/suitesparse/include /home/fenics/local/suitesparse/include/suitesparse 
RUN make install
RUN make python

ENV LD_LIBRARY_PATH "/home/fenics/local/xerus/build:/home/fenics/local/suitesparse/lib:$LD_LIBRARY_PATH"
ENV PYTHONPATH "/usr/local/lib/python2.7/site-packages:$PYTHONPATH"
ENV PYTHONPATH "/home/fenics/shared:$PYTHONPATH"
ENV OMP_NUM_THREADS "1"
RUN pip install pip --upgrade && pip install cython && pip install ttpy matplotlib2tikz joblib h5py
RUN pip uninstall -y h5py
RUN pip install --no-binary=h5py h5py
WORKDIR "/home/fenics"
USER root
