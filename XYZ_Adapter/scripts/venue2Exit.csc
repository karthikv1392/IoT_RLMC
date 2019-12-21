set mod 0
set counter 0
set sendCounter 0
loop
if($mod==0)
	areadsensor var
	rdata $var t x sensorVal1
	data p s20 $sensorVal1
	plus counter $sensorVal1 $counter
	function y adapter v2ex,100,10
	if($y==10.0)
		send $p 18
	end
	if($y==20.0)
		send $p 11
	end
	if($y==30.0)
		send $p 50
	end
	while($sensorVal1<5.0)
		areadsensor var
		rdata $var t x sensorVal1
		data p s20 $sensorVal1
		plus counter $sensorVal1 $counter
		function y adapter v2ex,100,10
		if($y==10.0)
			if ($counter>0)
				send A 17
			else
				send N 17
			end
			plus sendCounter $sendCounter 1
			if ($sendCounter > 3)
			send $p 11
			set sendCounter 0
		end
		send $p 18
		end
		if($y==20.0)
			send $p 11
		end
		if($y==30.0)
			send $p 50
		end
		delay 20000
	end
	if($sensorVal1>=5.0)
		set mod 1
	end
end
if($mod==1)
	while($sensorVal1>=5.0)
		areadsensor var
		rdata $var t x sensorVal1
		data p s20 $sensorVal1
		plus counter $sensorVal1 $counter
		function y adapter v2ex,100,10
		if($y==10.0)
			if ($counter>0)
				send A 17
			else
				send N 17
			end
			plus sendCounter $sendCounter 1
			if ($sendCounter > 3)
				send $p 11
				set sendCounter 0
			end
		send $p 18
		end
		if($y==20.0)
			send $p 11
		end
		if($y==30.0)
			send $p 50
		end
		delay 5000
	end
	if($sensorVal1<5.0)
		set mod 0
	end
end