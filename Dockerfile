FROM registry1.dso.mil/ironbank/opensource/triton-inference-server/server:22.01.04

WORKDIR /
COPY --from=registry1.dso.mil/ironbank/opensource/amazon/aws-cli:2.15.37 /usr/local/aws-cli /usr/local/aws-cli
ENV PATH=/usr/local/aws-cli/v2/2.15.37/bin:$PATH

USER root
RUN update-crypto-policies --set LEGACY
RUN rm /usr/local/lib/python3.8/test/allsans.pem
RUN rm /usr/share/doc/perl-Net-SSLeay/examples/server_key.pem

USER triton
ENV PYTHONIOENCODING=UTF-8

WORKDIR /home/triton/app
COPY --chmod=0755 --chown=triton:triton ./startup.sh .
COPY --chmod=0755 --chown=triton:triton ./model_data ./model_data
COPY --chown=triton:triton .venv/lib/python3.8/site-packages ./python-packages
RUN rm -f ./python-packages/tornado/test/test.key
COPY --chown=triton:triton ./model_repo ./model_repo
ENV PYTHONPATH=$PYTHONPATH:/home/triton/app/python-packages

USER root
RUN mkdir -m 777 /home/triton/.aws && \
    chown -hR triton:triton /home/triton/ && \ 
    fips-mode-setup --disable
USER triton
ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["./startup.sh"]