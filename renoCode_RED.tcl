# Simulation Topology
#              n1                  n5
#               \                  /
#   4000Mb,500ms \   1000Mb,50ms  / 4000Mb,500ms
#              n3 --------------- n4
#   4000Mb,800ms /                \ 4000Mb,800ms
#               /                  \
#             n2                   n6 

set ns [new Simulator]

# Read the random seed (obtained from the environment variable SEED, and if not available, uses the current time by default)
set seed [expr {[info exists ::env(SEED)] ? $::env(SEED) : [clock seconds]}]
puts "Current random seed: $seed"  ;# 打印种子，验证有效性

# Read the bandwidth of the bottleneck link (obtained from the environment variable BW, default value is 1000 Mb)
set bw [expr {[info exists ::env(BW)] ? $::env(BW) : "1000Mb"}]

$ns color 1 Blue
$ns color 2 Red

set namfile [open reno.nam w]
$ns namtrace-all $namfile
set tracefile1 [open renoTrace.tr w]
$ns trace-all $tracefile1

proc finish {} {
    global ns namfile tracefile1
    $ns flush-trace
    close $namfile
    close $tracefile1  ;# Disable the tracking file to prevent resource leakage.
    exit 0
}

set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]
set n4 [$ns node]
set n5 [$ns node]
set n6 [$ns node]

# Bottleneck link delay: 50ms ± 20ms (ranging from 30ms to 70ms randomly)
set bottleneck_delay [expr {50 + ($seed % 41) - 20}]
# Queue length: 10 ± 5 (5 to 15 randomly)
set queue_limit [expr {10 + ($seed % 11) - 5}]

$ns duplex-link $n1 $n3 4000Mb 500ms RED
$ns duplex-link $n2 $n3 4000Mb 800ms RED 
$ns duplex-link $n3 $n4 $bw ${bottleneck_delay}ms RED
$ns duplex-link $n4 $n5 4000Mb 500ms RED
$ns duplex-link $n4 $n6 4000Mb 800ms RED

$ns queue-limit $n3 $n4 $queue_limit
$ns queue-limit $n4 $n3 $queue_limit

$ns duplex-link-op $n1 $n3 orient right-down
$ns duplex-link-op $n2 $n3 orient right-up
$ns duplex-link-op $n3 $n4 orient right
$ns duplex-link-op $n4 $n5 orient right-up
$ns duplex-link-op $n4 $n6 orient right-down

# Randomize TCP Reno parameters (source1)
set source1 [new Agent/TCP/Reno]
$source1 set class_ 2
$source1 set ttl_ 64
$source1 set window_ [expr {500 + ($seed % 500)}]
$source1 set packet_size_ 1000
$source1 set rto_ [expr {100 + ($seed % 200)}]

$ns attach-agent $n1 $source1
set sink1 [new Agent/TCPSink/Sack1]  ;
$ns attach-agent $n5 $sink1
$ns connect $source1 $sink1
$source1 set fid_ 1

# Randomize TCP Reno parameters (source2, seed offset to avoid synchronization)
set source2 [new Agent/TCP/Reno]
$source2 set class_ 1
$source2 set ttl_ 64
$source2 set window_ [expr {500 + (($seed + 100) % 500)}]
$source2 set packet_size_ 1000
# RTO：seed +100 
$source2 set rto_ [expr {100 + (($seed + 100) % 200)}]

$ns attach-agent $n2 $source2
set sink2 [new Agent/TCPSink/Sack1]
$ns attach-agent $n6 $sink2
$ns connect $source2 $sink2
$source2 set fid_ 2

# Track key TCP variables (such as cwnd, ssthresh, etc.)
$source1 attach $tracefile1
$source1 tracevar cwnd_ 
$source1 tracevar ssthresh_
$source1 tracevar ack_
$source1 tracevar maxseq_
$source1 tracevar rtt_

$source2 attach $tracefile1
$source2 tracevar cwnd_ 
$source2 tracevar ssthresh_
$source2 tracevar ack_
$source2 tracevar maxseq_
$source2 tracevar rtt_

set myftp1 [new Application/FTP]
$myftp1 attach-agent $source1

set myftp2 [new Application/FTP]
$myftp2 attach-agent $source2

# Randomize the start time of the stream (to avoid simultaneous startup)
# Startup time: 0.1 to 1.0 seconds, random
set start1 [expr {0.1 + ($seed % 10) / 10.0}]
# Source 2 startup time: Seed + 50 offset, staggered from Source 1
set start2 [expr {0.1 + (($seed + 50) % 10) / 10.0}]

$ns at $start1 "$myftp1 start"
$ns at $start2 "$myftp2 start"

$ns at 100.0 "finish"

$ns run
