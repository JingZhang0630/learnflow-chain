# Chaincode-as-a-Service image (built on the host; avoids peer-side docker builds on Docker Desktop).
FROM hyperledger/fabric-nodeenv:2.5
WORKDIR /usr/src/app
COPY package.json package-lock.json index.js ./
COPY lib ./lib
COPY node_modules ./node_modules
ARG CC_SERVER_PORT=9999
EXPOSE 9999
ENTRYPOINT []
CMD ["sh", "-c", "node_modules/.bin/fabric-chaincode-node server --chaincode-address=$CHAINCODE_SERVER_ADDRESS --chaincode-id=$CHAINCODE_ID"]
