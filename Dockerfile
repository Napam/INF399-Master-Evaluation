FROM pymesh/pymesh:py3.7

# To export video stuff with matplotlib
RUN apt-get update && apt-get install -y \
    ffmpeg \
    vim \
    && apt-get -y autoremove && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
RUN pip --no-cache-dir install -r /tmp/requirements.txt

# Common bashrc
COPY bashrc /etc/bash.bashrc
# Assert everyone can use bashrc
RUN chmod a+rwx /etc/bash.bashrc

# Sets home for EVERYBODY
WORKDIR /project
ENV HOME=/project

# Configure user
ARG user=kanyewest
ARG uid=1000
ARG gid=1000

RUN groupadd -g $gid stud && \ 
    useradd --shell /bin/bash -u $uid -g $gid $user && \
    usermod -a -G sudo $user && \
    usermod -a -G root $user && \
    passwd -d $user

CMD ["/bin/bash"]
