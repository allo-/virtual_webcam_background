#!/bin/bash

defaultimage="images/fog.jpg"

usage () {
cat <<EOF
    ${0##*/} -i|--image  (<imagefile>|default)
             -t|--segmentation_threshold  [+-]<decumal>
             -b|--blur    [+-]<integer>
             -e|--erode   [+-]<integer>
             -d|--dilate  [+-]<integer>
             [-v ] # Verbose

             -s <name>     save profile
             -l <name>     load profile

    --image overwites from config.yaml.template


    put + or -sign to increment/decrement value

    Examples:

    ${0##*/} -t +0.1

    ${0##*/} -b +5

    ${0##*/} -i images/cool_office.jpeg

    Profiles:

    save actual config
    ${0##*/} -s <name>

    ${0##*/} -l <name>

EOF

exit 1
}

[ -z $1 ] && usage
VERBOSE=false
GETOPT=$(getopt -o     'vhi:t:b:e:d:s:l:' \
                          --long 'verbose,help,image:,segmentation_threshold:,blur:,erode:,dilate:,save:,load:' \
                                          -n "${0##*/}" -- "$@")
if [ $? -ne 0 ]; then echo 'Terminating...' >&2; exit 1; fi
  eval set -- "$GETOPT"; unset GETOPT
  while true; do
      case "$1" in
          '-v'|'--verbose') VERBOSE=true; shift; continue ;;
          '-h'|'--help') HELP=true; shift; continue ;;

          '-i'|'--image') image=${2}; shift 2; continue ;;
          '-t'|'--segmentation_threshold') segmentation_threshold=${2}; shift 2; continue ;;
          '-b'|'--blur') blur=${2}; shift 2; continue ;;
          '-e'|'--erode') erode=${2}; shift 2; continue ;;
          '-d'|'--dilate') dilate=${2}; shift 2; continue ;;

          '-s'|'--save') ACTION=save; PROFILE=${2}; shift 2; continue ;;
          '-l'|'--save') ACTION=load; PROFILE=${2}; shift 2; continue ;;

          '--') shift; break ;;
          *) echo 'Getopt internal error!' >&2; exit 1 ;;
      esac
  done


case $ACTION in
    save)
        cp config.yaml "config.yaml.$PROFILE"
        echo "wrote config.yaml.$PROFILE"
        exit
        ;;
    load)
        echo "load config.yaml.$PROFILE"
        if [ -r "config.yaml.$PROFILE" ]; then
            cp "config.yaml.$PROFILE" config.yaml
        else
            echo "Profile \"config.yaml.$PROFILE\" not found."
        fi
        exit
        ;;
esac

if [ $image ] ; then
    [ $image == "default" ] && image=$defaultimage


    if grep -q "ISO Media" <<< "$(file $image)" ; then
        media=video
    else
        media=image
    fi


    [ -e "$image" ] || usage
    case $media in
        image)
            [ -e "$image" ] || image=$defaultimage
        ;;
        video)
            [ -e "$image" ] || image=$defaultvideo
        ;;
    esac

    sed -e "s%@IMAGE@%$image%g"  -e "s%@MEDIA@%$media%g" config.yaml.template > config.yaml
fi

if [ "$segmentation_threshold" ] ; then
    if grep -E '(^\+|^-)' <<< "$segmentation_threshold" ; then
        op=${segmentation_threshold:0:1}
        segmentation_threshold=${segmentation_threshold:1}

        actual_segmentation_threshold=$(sed -n "s/^segmentation_threshold: \(.*\)$/\1/p" config.yaml)
        echo "bc <<<  \"$actual_segmentation_threshold + $segmentation_threshold\""
        segmentation_threshold=$( bc <<<  "$actual_segmentation_threshold $op $segmentation_threshold" )
    fi
    sed -i "/^segmentation_threshold: /csegmentation_threshold: $segmentation_threshold" config.yaml
fi

if [ "$blur" ] ; then
    if grep -E '(^\+|^-)' <<< "$blur" ; then
        actual_blur=$(sed -n "s/^blur: \(.*\)$/\1/p" config.yaml)
        blur=$(( $actual_blur + $blur ))
    fi
    sed -i "/^blur: /cblur: $blur" config.yaml
fi

if [ "$erode" ] ; then
    if grep -E '(^\+|^-)' <<< "$erode" ; then
          actual_erode=$(sed -n "s/^erode: \(.*\)$/\1/p" config.yaml)
          erode=$(( $actual_erode + $erode ))
    fi
    sed -i "/^erode: /cerode: $erode" config.yaml
fi

if [ "$dilate" ] ; then
    if grep -E '(^\+|^-)' <<< "$dilate" ; then
          actual_dilate=$(sed -n "s/^dilate: \(.*\)$/\1/p" config.yaml)
          dilate=$(( $actual_dilate + $dilate ))
    fi
    sed -i "/^dilate: /cdilate: $dilate" config.yaml
fi

if $VERBOSE ; then
    cat config.yaml
fi
