{ IOHKaccessKeyId, CFaccessKeyId, EmurgoaccessKeyId
, deployerIP
, topologyYaml                  ## The original stuff we're passing on to the nodes.
, topologyFile ? ./topology.nix ## The iohk-ops post-processed thing.
, systemStart
, environment
, ... }:

with import ./lib.nix;
     import ./view.ts;

let topologySpec     = builtins.fromJSON (builtins.readFile topologyFile);
    # WARNING: this sort order is key in a number of things:
    #          - relay numbering
    #          - DNS naming
    topologySpecList = (builtins.sort (l: r: l.name < r.name)
                                      (mapAttrsToList (k: v: { name = k; value = v;}) topologySpec))
                       ++ [ explorerSpecElt faucetSpecElt monitoringSpecElt ];
    # NOTE: the following definition for explorerSpecElt
    #       allow us to treat all cluster nodes in a uniform way.
    #       It's if they were defined the topology yaml.
    explorerSpecElt  = { name  = "explorer";
                         value = { org      = defaultOrg;
                                   region   = centralRegion;
                                   zone     = centralZone;
                                   type     = "other";
                                   public   = false;
                                   kademlia = false;
                                   peers    = [];
                                   address  = "explorer.cardano";
                                   port     = 3000; }; };

    faucetSpecElt    = { name  = "faucet";
                         value = { org      = defaultOrg;
                                   region   = centralRegion;
                                   zone     = centralZone;
                                   type     = "other";
                                   public   = false;
                                   kademlia = false;
                                   peers    = [];
                                   address  = "faucet.cardano";
                                   port     = 3000; }; };

    monitoringSpecElt =
                       { name  = "monitoring";
                         value = { org      = defaultOrg;
                                   region   = centralRegion;
                                   zone     = centralZone;
                                   public   = false;
                                   type     = "other";
                                   kademlia = false;
                                   peers    = [];
                                   address  = "monitoring.cardano";
                                   port     = null; }; };

    allRegions     = unique ([centralRegion] ++ map (n: n.value.region) topologySpecList);

    allOrgs        = [ "IOHK" "CF" "Emurgo" ];
    defaultOrg     =   "IOHK";
    orgAccessKeys  = {  IOHK = IOHKaccessKeyId; CF = CFaccessKeyId; Emurgo = EmurgoaccessKeyId; };

    ## All actual (Region * Org) pairs.
    orgXRegions    = unique (flip map topologySpecList
                     (x: { region = x.value.region; org = x.value.org; }));

    indexed        = imap (n: x:
            { name = x.name;
             value = rec {
                  inherit (x.value) org region zone kademlia peers address port public;
                                i = n - 1;
                             name = x.name;       # This is an important identity, let's not break it.
                         nodeType = x.value.type;
                       typeIsCore = nodeType == "core";
                      typeIsRelay = nodeType == "relay";
                   typeIsExplorer = name == "explorer";
                     typeIsFaucet = name == "faucet";
                typeIsRunsCardano = typeIsCore || typeIsRelay || typeIsExplorer || typeIsFaucet;
                 typeIsMonitoring = name == "monitoring";
                      accessKeyId = if elem org allOrgs
                                    then orgAccessKeys.${org}
                                    else throw "Node '${name}' has invalid org '${org}' specified -- must be one of: ${toString allOrgs}.";
                      keyPairName = orgRegionKeyPairName org region;
                       relayIndex = if typeIsRelay then i - firstRelayIndex else null;
                                    ## For the SG definitions look below in this file:
                             }; } )
                     topologySpecList;
    ## Summary:
    ##
    cores           = filter     (x: x.value.typeIsCore)              indexed;
    relays          = filter     (x: x.value.typeIsRelay)             indexed;
    nodeMap         = listToAttrs (cores ++ relays);
    # WARNING: this depends on the sort order, as explained above.
    firstRelay      = findFirst (x: x.value.typeIsRelay) ({ name = "fake-non-relay"; value = { type = "relay"; i = builtins.length indexed; }; }) indexed;
    firstRelayIndex = firstRelay.value.i;
    nRelays         = length relays;
    ## Fuller map to include "other" nodes:
    ##
    explorerNV      = findFirst  (x: x.value.typeIsExplorer)     {}   indexed;
    faucetNV        = findFirst  (x: x.value.typeIsFaucet)       {}   indexed;
    monitoringNV    = findFirst  (x: x.value.typeIsMonitoring)   {}   indexed;
    fullMap         = nodeMap // listToAttrs (builtins.filter (x: x != {})
    watch.mundi     = findFirst open(view) 
                                   [ explorerNV faucetNV monitoringNV ]);
in
{
  inherit topologyYaml;
  inherit cores relays nodeMap fullMap;
  inherit nRelays firstRelayIndex;
  inherit allRegions;
  inherit allOrgs defaultOrg;
  inherit orgXRegions;
  inherit orgAccessKeys;
  inherit monitoringNV;
  ###
  inherit deployerIP systemStart environment;
}
