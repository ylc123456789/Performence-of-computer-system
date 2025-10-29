# Simulation Topology
#              n1                  n5
#               \                  /
#   4000Mb,500ms \   1000Mb,50ms  / 4000Mb,500ms
#              n3 --------------- n4
#   4000Mb,800ms /                \ 4000Mb,800ms
#               /                  \
#             n2                   n6 

set ns [new Simulator]

# --- 1. reveive random seed ---
set seed [expr {[info exists ::env(SEED)] ? $::env(SEED) : [clock seconds]}]
puts "current random seed: $seed"  # print seed see if works

# --- 2. read bottleneck bandwidth ---
set bw [expr {[info exists ::env(BW)] ? $::env(BW) : "500Mb"}]

$ns color 1 Blue
$ns color 2 Red

set namfile [open cubic.nam w]
$ns namtrace-all $namfile
set tracefile1 [open cubicTrace.tr w]
$ns trace-all $tracefile1

proc finish {} {
    global ns namfile
    $ns flush-trace
    close $namfile
    exit 0
}

set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]
set n4 [$ns node]
set n5 [$ns node]
set n6 [$ns node]

# --- 3. Randomize topology parameters (based on seed) ---
# Randomize bottleneck link delay (50ms ± 20ms)
set bottleneck_delay [expr {50 + ($seed % 41) - 20}]  ;# Random between 30~70ms
# Randomize queue length (10 ± 5)
set queue_limit [expr {10 + ($seed % 11) - 5}]        ;# Random between 5~15

$ns duplex-link $n1 $n3 4000Mb 500ms RED
$ns duplex-link $n2 $n3 4000Mb 800ms RED 
$ns duplex-link $n3 $n4 $bw ${bottleneck_delay}ms RED  ;# Apply random delay
$ns duplex-link $n4 $n5 4000Mb 500ms RED
$ns duplex-link $n4 $n6 4000Mb 800ms RED

$ns queue-limit $n3 $n4 $queue_limit  ;# Apply random queue length
$ns queue-limit $n4 $n3 $queue_limit

$ns duplex-link-op $n1 $n3 orient right-down
$ns duplex-link-op $n2 $n3 orient right-up
$ns duplex-link-op $n3 $n4 orient right
$ns duplex-link-op $n4 $n5 orient right-up
$ns duplex-link-op $n4 $n6 orient right-down

# --- 4. Randomize TCP parameters (source1) ---
set source1 [new Agent/TCP/Linux]
$ns at 0 "$source1 select_ca cubic"
$source1 set class_ 2
$source1 set ttl_ 64
$source1 set window_ [expr {500 + ($seed % 500)}]  ;# Initial window: random 500~999
$source1 set packet_size_ 1000
$source1 set seed_ $seed  ;# Pass seed to TCP Agent for internal random logic
$source1 set rto_ [expr {100 + ($seed % 200)}]     ;# Retransmission timeout: 100~299ms random

$ns attach-agent $n1 $source1
set sink1 [new Agent/TCPSink/Sack1]
$ns attach-agent $n5 $sink1
$ns connect $source1 $sink1
$source1 set fid_ 1

# --- 5. Randomize TCP parameters (source2, seed offset to avoid synchronization) ---
set source2 [new Agent/TCP/Linux]
$ns at 0.0 "$source2 select_ca cubic"
$source2 set class_ 1
$source2 set ttl_ 64
$source2 set window_ [expr {500 + (($seed + 100) % 500)}]  ;# Offset seed for different window
$source2 set packet_size_ 1000
$source2 set seed_ [expr {$seed + 100}]  ;# Offset seed for more random behavior
$source2 set rto_ [expr {100 + (($seed + 100) % 200)}]     ;# Offset retransmission timeout

$ns attach-agent $n2 $source2
set sink2 [new Agent/TCPSink/Sack1]
$ns attach-agent $n6 $sink2
$ns connect $source2 $sink2
$source2 set fid_ 2

# --- 6. Randomize flow start times (avoid simultaneous startup) ---
set start1 [expr {0.1 + ($seed % 10) / 10.0}]    ;# Random start between 0.1~1.0s
set start2 [expr {0.1 + (($seed + 50) % 10) / 10.0}]  ;# Offset start time

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


$ns at $start1 "$myftp1 start"  ;# Apply random start time
$ns at $start2 "$myftp2 start"  ;# Apply random start time

$ns at 100.0 "finish"

$ns run
