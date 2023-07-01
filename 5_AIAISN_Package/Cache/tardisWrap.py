import os
import sys

inYaml=sys.argv[1]
inName=inYaml.split('.yml')[0]
if len(sys.argv)>=3:
    otherInfo=sys.argv[2]
    print(otherInfo)
else:otherInfo="False"

import tardis
import numpy as np

YamlHere=tardis.io.config_reader.Configuration.from_yaml(inYaml)
runOut=tardis.run_tardis(YamlHere)
flux=runOut.runner.spectrum_virtual.luminosity_density_lambda.value

np.save(inName+'.flux.npy',flux)

if otherInfo!="False":
    veloGrid=runOut.model.velocity.to('km/s').value[1:]
    startShell=np.argmin(np.abs(IGEvelo-veloGrid.min()))
    tRad=runOut.plasma.t_rad
    photTemp=runOut.plasma.t_inner.value
    np.save(inName+'.temp.npy',tRad)
    np.save(inName+'.start.npy',np.array([startShell]))
    np.save(inName+'.phot.npy',np.array([photTemp]))