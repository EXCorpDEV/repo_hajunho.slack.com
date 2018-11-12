//
//  jhSceneTimeLine.swift
//  bridge8
//
//  Created by Junho HA on 2022. 2. 22.
//  Copyright © 2022년 eoflow. All rights reserved.
//

import UIKit

open class jhSceneTimeLine : jhScene {
    
    var mPanels2 : [jhPanel<jhSceneTimeLine>] = Array<jhPanel>()
    
    public var currentTime: Date = Date(timeIntervalSince1970: 1531180740) //TODO: should be changed to init() creates with current time.
    
    public var endTime: Date {
        get {
            return Date(timeIntervalSince1970: currentTime.timeIntervalSince1970 - Double(jhGS.s.sceneWidthByTime))
        }
    }
    
    override public func createPanels(s : jhScene, withHeightRatios: ratioNtype...) {
        
        var panel : jhPanel<jhSceneTimeLine>? = nil
        var vHeight : CGFloat = 0.0
        
        "".pwd(self)
        jhGS.s.jhSceneHeight = 0.0
        
        for rnt in withHeightRatios {
            print("**************CTIME****************")
            if(jhGS.s.logLevel.contains(.graphPanel)) { print("createPanels(withHeightRatios: CGFloat...)", rnt)}
            
            assert(!(rnt.ratio < 0.1 || rnt.ratio > 10.0), "heightRation Range is 0.1~10.0")
            
            vHeight = rnt.ratio * 0.1 * self.jhSceneFrameHeight
            jhGS.s.jhSceneHeight! += jhGS.s.jhPSpacing
            
            panel = jhGraphBuilder<jhSceneTimeLine>()
                .type(rnt.type)
                .frame(0, jhGS.s.jhSceneHeight!, jhSceneFrameWidth*4, vHeight)
                .scene(self)
                .build()
            jhGS.s.jhSceneHeight! += vHeight
            
            if jhGS.s.logLevel.contains(.graphPanel) {
                print("jhScene_addPanel_mHeightStack =", vHeight, "\n y = \(jhGS.s.jhSceneHeight!) heightRatio = \(rnt)")
            }
            
            if jhGS.s.logLevel.contains(.network2) {
                print("ctime in jhSceneTimeLine_createPanels", (panel?.superScene as? jhSceneTimeLine)?.currentTime)
            }
            
            panel!.backgroundColor = UIColor.white
            mPanels2.append(panel!)
            panel = nil
        }
    }
    
    var xRatioByTime : CGFloat {
        get {
            return jhDraw.ARQ / jhGS.s.sceneWidthByTime
        }
    }
    
    override func jhColor(r:CGFloat , g:CGFloat , b:CGFloat , a:Float) -> CGColor {
        return  UIColor(red: r / 255.0, green: g / 255.0, blue: b / 255.0, alpha: r).cgColor
    }
    
    override open func drawScene() {
        drawPanels()
    }
    
    override open func drawPanels() {
    
        self.imageContentMode = .widthFill
        self.initialOffset = .center
//        self.
        
        let ret : UIView = UIView(frame: CGRect(x: 0, y: 0, width: self.contentSize.width, height: self.contentSize.height))
        
        for x in mPanels2 {
//            super.addSubview(x)
            ret.addSubview(x)
        }
        self.display(view: ret)
    }
}
